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
    return 0;
}
