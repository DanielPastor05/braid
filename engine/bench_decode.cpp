// What a KV cache could win, measured before one is built.
//
// READ THE LAST PARAGRAPH FIRST. The headline number below is measured against a
// baseline this server stopped running the day after: it computes 256 positions
// per step, and the worker now runs at the width of its longest row. The table
// is still true about window widths and still worth having. What it is not is
// what a cache would buy here.
//
// Serving this model used to cost one forward pass over all 256 positions per
// token, with the sampler reading exactly one row of the result. A KV cache
// would keep the keys and values of the positions already seen and run the
// projections and the feed-forward over the single new position instead of over
// all of them -- which on paper is a 256x reduction in arithmetic, because
// attention is only about a tenth of the FLOPs at this geometry and everything
// else is per-position.
//
// On paper, and it is not what happens. This measures the ceiling without
// building the cache -- the same model and weights over a window of S positions
// instead of 256 -- and the answer is that the win is bounded by a floor that
// has nothing to do with arithmetic:
//
//   batch |  S=256  |  S<=64  |  ceiling
//       1 | 6.41 ms | 2.33 ms |    2.7x
//       8 | 21.3 ms | 2.30 ms |    9.3x
//      32 | 83.8 ms | 2.52 ms |   33.3x
//
// The floor is the same ~2.4 ms at every batch size, because it is 177 kernel
// launches and a launch does not care how much work it carries: 2.4 ms / 177 is
// 13 microseconds each. A cache makes the kernels smaller, not fewer, so the
// naive 256x is not available at any batch size.
//
// Run it with the engine's *own* dispatch defaults and the win is smaller still
// -- 1.6x at a batch of one -- because a narrow window falls below the floors
// that keep small work on the CPU, and the sweep stops comparing window sizes
// and starts comparing execution paths. Which is to say a KV cache here would
// re-activate the three threshold pull requests this project already landed
// upstream, for a different reason than the first time.
//
// A cached decode is not exactly a forward at S=1: its attention still reads the
// 256 cached keys this does not. That read is about 0.4 ms at a batch of 32
// against a 2.5 ms floor, so the numbers above are a ceiling and a close one.
//
//   braid_bench_decode <model-prefix> [repeats]
//
// ENGINE_CUDA_MIN_FLOPS, ENGINE_CUDA_MIN_ELEMENTS and ENGINE_CUDA_MIN_LAYERNORM
// are read by the engine at start; set them to 1 for the path-consistent sweep.
//
// The correction, kept here rather than in a commit message because it is the
// point. Those ceilings divide a 256-position step by a 1-position step, and the
// worker stopped running 256-position steps: it runs at the width of the longest
// row in the batch, whose mean the server reports as 29 under the load harness.
// The remaining win is 29 -> 1, which this table puts at roughly 4x at a batch of
// thirty-two rather than 33x. A ratio is only as honest as its denominator, and
// that denominator was itself waste.
//
// It grows with the generation. A server whose clients ask for a thousand tokens
// would run near the full context and get most of the 33x back. This one asks
// for thirty.
//
// And the cache cannot be used here in any case: it is keyed by position, one
// write offset is shared by the whole batch, and only about 2% of steps have
// every row at the same position. Continuous batching is the practice of making
// sure they do not. The server counts it as aligned_step_share.

#include "charmodel.hpp"

#include "engine/autograd.hpp"
#include "engine/cuda.hpp"
#include "engine/data.hpp"
#include "engine/nn.hpp"
#include "engine/serialize.hpp"
#include "engine/tensor.hpp"

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

double ms_since(clock_type::time_point from) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(clock_type::now() - from).count() /
           1e6;
}

// One forward over a window of `seq` positions. This is CharModel::forward with
// the positional table and the mask cut to the same width, which is the only
// thing that stops the model insisting on all 256.
engine::Tensor forward_at(braid::CharModel& model, const engine::Tensor& ids,
                          const engine::Tensor& positions, const engine::Tensor& mask) {
    engine::Tensor h = model.embedding.forward(ids) + positions;
    for (engine::nn::TransformerBlock& block : model.blocks) h = block.forward(h, &mask);
    return model.head.forward(h);
}

struct Row {
    std::size_t seq = 0;
    double ms = 0.0;
    double kernels = 0.0;
    double to_device = 0.0;
    double to_host = 0.0;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: braid_bench_decode <model-prefix> [repeats]\n";
        return 2;
    }
    const std::string prefix = argv[1];
    const int repeats = argc > 2 ? std::atoi(argv[2]) : 20;
    if (repeats < 1) {
        std::cerr << "repeats must be at least 1\n";
        return 2;
    }

    const std::string alphabet = read_file(prefix + ".vocab");
    if (alphabet.empty()) {
        std::cerr << "could not read " << prefix << ".vocab\n";
        return 1;
    }
    const engine::data::CharVocab vocab(alphabet);

    braid::CharModel model(vocab.size());
    try {
        auto named = model.named_parameters();
        engine::load_parameters(named, prefix + ".bin");
    } catch (const std::exception& e) {
        std::cerr << "could not load " << prefix << ".bin: " << e.what() << "\n";
        return 1;
    }
    model.train(false);

    std::cerr << "braid_bench_decode: vocab " << vocab.size() << ", cuda "
              << (engine::cuda::available() ? "yes" : "no") << ", " << repeats << " repeats\n";

    engine::autograd::NoGradGuard no_grad;

    // Out to the full 1024-id context. The list stopped at 256 when the model
    // did, which made the last column a ratio against a width the server can now
    // exceed by four times.
    const std::size_t widths[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};

    // "vs full" rather than "vs 256": the ratio divides by the last width in the
    // sweep, so it followed the context out to 1024 on its own and the header
    // did not.
    std::cout << "| batch | window | forward ms | kernels | to_device | to_host | vs full |\n";
    std::cout << "|---|---|---|---|---|---|---|\n";

    for (std::size_t n : {std::size_t{1}, std::size_t{8}, std::size_t{16}, std::size_t{32}}) {
        std::vector<Row> rows;
        for (std::size_t seq : widths) {
            // The window's positional slice and its mask, built once. The worker
            // builds its mask once at start too, so charging their construction
            // to a step would measure this benchmark rather than the model.
            const engine::Tensor positions = model.positions.slice(0, 0, seq);
            const engine::Tensor mask = engine::nn::causal_mask(seq);

            engine::Tensor ids({n, seq}, 0.0f, false);
            float* values = ids.data();
            for (std::size_t i = 0; i < n * seq; ++i) {
                values[i] = static_cast<float>(i % vocab.size());
            }

            // Warm, so the first pass's allocations and lazy device buffers are
            // not charged to the sample.
            for (int i = 0; i < 3; ++i) {
                const engine::Tensor logits = forward_at(model, ids, positions, mask);
                engine::cuda::synchronize();
            }

            const std::size_t kernels_before = engine::cuda::kernels_launched();
            const auto transfers_before = engine::cuda::transfer_stats();
            const auto started = clock_type::now();
            for (int i = 0; i < repeats; ++i) {
                const engine::Tensor logits = forward_at(model, ids, positions, mask);
                // The kernels are asynchronous: without this, the loop would
                // time how fast they can be launched.
                engine::cuda::synchronize();
            }
            const double per_step = ms_since(started) / repeats;
            const auto transfers_after = engine::cuda::transfer_stats();

            Row row;
            row.seq = seq;
            row.ms = per_step;
            row.kernels =
                static_cast<double>(engine::cuda::kernels_launched() - kernels_before) / repeats;
            row.to_device =
                static_cast<double>(transfers_after.to_device_count - transfers_before.to_device_count) /
                repeats;
            row.to_host =
                static_cast<double>(transfers_after.to_host_count - transfers_before.to_host_count) / repeats;
            rows.push_back(row);
        }

        // Printed after the sweep because every row is quoted against the full
        // window, which is the last one measured.
        const double at_full = rows.back().ms;
        for (const Row& row : rows) {
            std::cout << "| " << n << " | " << row.seq << " | " << std::fixed
                      << std::setprecision(3) << row.ms << " | " << std::setprecision(0)
                      << row.kernels << " | " << row.to_device << " | " << row.to_host << " | "
                      << std::setprecision(1) << (row.ms > 0.0 ? at_full / row.ms : 0.0) << "x |\n";
        }
        std::cout << "|---|---|---|---|---|---|---|\n";
        std::cout.flush();
    }
    return 0;
}
