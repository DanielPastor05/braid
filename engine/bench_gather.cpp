// What it costs to gather the active slots of a key/value cache.
//
// A slot-indexed cache lives on the card at (slots, heads, capacity, head_dim)
// and persists between steps. The batch being stepped is some subset of those
// slots in some order, and the engine's cached forward wants a cache whose row
// count matches the batch -- so getting from one to the other is a gather, and
// putting the result back is a scatter.
//
// That is the design the plan calls for, and this measures its price before
// anything is built on it, because the arithmetic is not obviously favourable.
// One block's cache at 64 slots, 6 heads, 1024 positions and 64 head_dim is
// 100 MB; keys and values make it 200 MB; six blocks make it 1.2 GB. Moving
// that twice a step, on a card whose measured copy bandwidth is what it is,
// wants comparing against a cached step of roughly three milliseconds.
//
// The comparison this prints is deliberately generous to the gather: it charges
// only the read, not the write back, and it gathers a contiguous prefix rather
// than the scattered order a real batch produces.
//
//   braid_bench_gather [repeats]

#include "engine/cuda.hpp"
#include "engine/tensor.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

double ms_since(const clock_type::time_point& start) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
}

// The geometry braid actually serves at: 6 blocks of 6 heads, 64 per head.
constexpr std::size_t kHeads = 6;
constexpr std::size_t kHeadDim = 64;
constexpr std::size_t kBlocks = 6;

}  // namespace

int main(int argc, char** argv) {
    const int repeats = argc > 1 ? std::atoi(argv[1]) : 20;

    if (!engine::cuda::available()) {
        std::cerr << "no CUDA device; this benchmark is about device-side movement\n";
        return 1;
    }
    std::cerr << "braid_bench_gather: " << repeats << " repeats\n";

    std::cout << "| slots | taken | positions | one block MB | gather ms | "
                 "whole model ms | GB/s |\n";
    std::cout << "|---|---|---|---|---|---|---|\n";

    for (std::size_t slots : {std::size_t{16}, std::size_t{32}, std::size_t{64}}) {
        for (std::size_t positions : {std::size_t{128}, std::size_t{453}, std::size_t{1024}}) {
            // One block's keys, as the cache would hold them: a row per slot,
            // flattened, because select_rows gathers whole rows of the first
            // axis and that is exactly what a slot is.
            const std::size_t row = kHeads * positions * kHeadDim;
            engine::Tensor cache({slots, row}, 0.0f, false);

            // Make it device-resident before timing anything.
            //
            // This is not decoration. `select_rows` admits the device path on
            // `resident_on_device()`, so a tensor that has only ever lived on the
            // host silently takes the host path -- and the first version of this
            // benchmark did exactly that and reported 3-9 GB/s, which is PCIe
            // and not the card. An elementwise op with the dispatch floors at 1
            // is what puts it on the device and leaves it there.
            {
                const engine::Tensor onto_the_card = cache + cache;
                engine::cuda::synchronize();
            }

            std::vector<std::size_t> take(slots);
            std::iota(take.begin(), take.end(), 0u);

            for (int i = 0; i < 3; ++i) {
                const engine::Tensor gathered = cache.select_rows(take);
                engine::cuda::synchronize();
            }

            const auto started = clock_type::now();
            for (int i = 0; i < repeats; ++i) {
                const engine::Tensor gathered = cache.select_rows(take);
                engine::cuda::synchronize();
            }
            const double per_gather = ms_since(started) / repeats;

            // Keys and values, every block. The read side only.
            const double one_block_mb =
                static_cast<double>(slots * row * sizeof(float)) / (1024.0 * 1024.0);
            const double whole_model = per_gather * 2.0 * kBlocks;
            const double gigabytes = static_cast<double>(slots * row * sizeof(float)) / 1e9;
            const double bandwidth = gigabytes / (per_gather / 1e3);

            std::cout << "| " << slots << " | " << slots << " | " << positions << " | "
                      << std::fixed << std::setprecision(1) << one_block_mb << " | "
                      << std::setprecision(3) << per_gather << " | " << std::setprecision(2)
                      << whole_model << " | " << std::setprecision(0) << bandwidth << " |\n";
        }
    }

    std::cout << "\nThe last column is the read alone, one block. `whole model ms` is that "
                 "times two for keys and values and times six for the blocks -- still only "
                 "the read, and still in slot order rather than a real batch's.\n";

    // ---- getting a *narrow* cache out of a wide pool ---------------------
    //
    // A cached step costs what its capacity is, so the compact cache handed to
    // the model should be as narrow as the batch's history requires -- rounded
    // up to a block, not up to the context. The pool, though, has to be as wide
    // as the context, because a sequence may run that far.
    //
    // So the per-step read is "the active slots, but only their first W
    // positions", and the engine has no such primitive. It has whole-row gather
    // and it has slice, and they compose two ways round with very different
    // amounts of copying:
    //
    //   slice the pool to W, then gather n of its rows: touches all `slots`
    //   gather n whole rows, then slice them to W:      touches the full context
    //
    // Which is cheaper depends on n/slots against W/context, and both ratios are
    // real at serving. Measured rather than reasoned about, because the arithmetic
    // for this has been wrong twice today.
    std::cout << "\n| slots | active | context | width | slice+gather ms | gather+slice ms |\n";
    std::cout << "|---|---|---|---|---|---|\n";

    constexpr std::size_t kContext = 1024;
    for (std::size_t slots : {std::size_t{32}, std::size_t{64}}) {
        for (std::size_t active : {std::size_t{8}, std::size_t{16}, std::size_t{32}}) {
            if (active > slots) continue;
            for (std::size_t width : {std::size_t{144}, std::size_t{464}}) {
                // The pool as a slot-indexed cache holds it: a row per slot,
                // (heads, context, head_dim) flattened behind it. Kept as
                // (slots, heads, context, head_dim) so axis 2 is the position
                // and slice can narrow it.
                engine::Tensor pool({slots, kHeads, kContext, kHeadDim}, 0.0f, false);
                {
                    const engine::Tensor onto_the_card = pool + pool;
                    engine::cuda::synchronize();
                }
                std::vector<std::size_t> take(active);
                for (std::size_t i = 0; i < active; ++i) take[i] = (i * 7) % slots;

                for (int i = 0; i < 3; ++i) {
                    (void)pool.slice(2, 0, width).select_rows(take);
                    (void)pool.select_rows(take).slice(2, 0, width);
                    engine::cuda::synchronize();
                }

                auto started = clock_type::now();
                for (int i = 0; i < repeats; ++i) {
                    (void)pool.slice(2, 0, width).select_rows(take);
                    engine::cuda::synchronize();
                }
                const double slice_first = (ms_since(started) / repeats) * 2.0 * kBlocks;

                started = clock_type::now();
                for (int i = 0; i < repeats; ++i) {
                    (void)pool.select_rows(take).slice(2, 0, width);
                    engine::cuda::synchronize();
                }
                const double gather_first = (ms_since(started) / repeats) * 2.0 * kBlocks;

                std::cout << "| " << slots << " | " << active << " | " << kContext << " | " << width
                          << " | " << std::fixed << std::setprecision(2) << slice_first << " | "
                          << gather_first << " |\n";
                std::cout.flush();
            }
        }
    }
    std::cout << "\nBoth columns are keys and values for every block, which is what one step "
                 "of a slot-indexed cache would pay before the model runs.\n";
    return 0;
}
