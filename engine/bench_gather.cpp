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
//   ENGINE_CUDA_MIN_FLOPS=1 ENGINE_CUDA_MIN_ELEMENTS=1 ENGINE_CUDA_MIN_LAYERNORM=1 \
//       braid_bench_gather [repeats]
//
// **All three, every time.** Every device path here is admitted on residency, and
// a tensor only becomes resident by having a kernel write it -- which is what the
// `x + x` before each timing loop is for. Leave one floor at its default and that
// elementwise add stays on the host, the tensor never reaches the card, and every
// operation below silently measures the host path instead.
//
// This has produced three wrong tables in one afternoon. The first read 3-9 GB/s
// for a gather, which is PCIe rather than a card with 448 GB/s of bandwidth; the
// last made a 192-launch write-back look nine times *faster* than the single
// kernel that replaces it. Both looked plausible. If a number here is within an
// order of magnitude of PCIe, check the environment before believing it.

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

    // ---- and putting the answer back ------------------------------------
    //
    // After the step, each row's new key and value have to reach the slot they
    // came from, at that row's own position. That is a gather's inverse and the
    // engine has neither half of it in one call: select_rows takes indices and
    // no offset, copy_into_rows takes an offset per row and no indices.
    //
    // So the write-back is one copy_into_rows per slot, per block, per tensor --
    // 192 calls at a batch of sixteen -- each moving heads * head_dim floats,
    // which is nothing. The question is whether the launches cost more than the
    // bytes, and at this size they plainly will; what matters is how much,
    // against a cached step of three to four milliseconds.
    std::cout << "\n| active | one-at-a-time | ms | per write us | scatter_rows ms | saved |\n";
    std::cout << "|---|---|---|---|---|---|\n";

    for (std::size_t active : {std::size_t{1}, std::size_t{8}, std::size_t{16}, std::size_t{32}}) {
        // One tensor per slot, which is what makes copy_into_rows usable at all:
        // its offsets index axis 0, so a per-slot cache has the one row it needs.
        std::vector<engine::Tensor> slots_k;
        slots_k.reserve(active);
        for (std::size_t r = 0; r < active; ++r) {
            engine::Tensor one({1, kHeads, kContext, kHeadDim}, 0.0f, false);
            const engine::Tensor onto_the_card = one + one;
            engine::cuda::synchronize();
            slots_k.push_back(one);
        }
        engine::Tensor fresh({1, kHeads, 1, kHeadDim}, 1.0f, false);
        {
            const engine::Tensor onto_the_card = fresh + fresh;
            engine::cuda::synchronize();
        }

        const std::size_t writes = active * 2 * kBlocks;
        for (int i = 0; i < 3; ++i) {
            for (std::size_t w = 0; w < writes; ++w) {
                slots_k[w % active].copy_into_rows(fresh, 2, {17});
            }
            engine::cuda::synchronize();
        }

        const auto started = clock_type::now();
        for (int i = 0; i < repeats; ++i) {
            for (std::size_t w = 0; w < writes; ++w) {
                slots_k[w % active].copy_into_rows(fresh, 2, {17});
            }
            engine::cuda::synchronize();
        }
        const double per_step = ms_since(started) / repeats;

        // The same write-back through scatter_rows: twelve launches instead of
        // `writes` of them, carrying exactly the same bytes.
        engine::Tensor pool({active, kHeads, kContext, kHeadDim}, 0.0f, false);
        engine::Tensor batch({active, kHeads, 1, kHeadDim}, 1.0f, false);
        {
            const engine::Tensor a = pool + pool;
            const engine::Tensor b = batch + batch;
            engine::cuda::synchronize();
        }
        std::vector<std::size_t> into(active), at(active);
        for (std::size_t r = 0; r < active; ++r) {
            into[r] = r;
            at[r] = 17 + r;
        }
        for (int i = 0; i < 3; ++i) {
            for (std::size_t b = 0; b < 2 * kBlocks; ++b) pool.scatter_rows(batch, 2, into, at);
            engine::cuda::synchronize();
        }
        const auto scatter_started = clock_type::now();
        for (int i = 0; i < repeats; ++i) {
            for (std::size_t b = 0; b < 2 * kBlocks; ++b) pool.scatter_rows(batch, 2, into, at);
            engine::cuda::synchronize();
        }
        const double with_scatter = ms_since(scatter_started) / repeats;

        std::cout << "| " << active << " | " << writes << " | " << std::fixed
                  << std::setprecision(3) << per_step << " | " << std::setprecision(1)
                  << (per_step * 1000.0 / static_cast<double>(writes)) << " | "
                  << std::setprecision(3) << with_scatter << " | " << std::setprecision(1)
                  << (per_step / (with_scatter > 0.0 ? with_scatter : 1.0)) << "x |\n";
        std::cout.flush();
    }
    return 0;
}
