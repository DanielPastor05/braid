// One braid worker: a process that owns the model and answers step requests
// over stdin and stdout.
//
// It is a separate process rather than a library the server links, and that was
// not a stylistic choice. nvcc on Windows compiles only through MSVC, and cgo
// links only through a GCC-compatible toolchain; a C++ static library from one
// is not linkable by the other. A pipe has no ABI to disagree about.
//
// The cost is one round trip per step -- serialise n windows, read n ids back --
// which the load harness measures against the GPU time rather than assuming it
// is negligible. The benefit is that a worker is a thing that can be killed,
// which is what serving on more than one of them will need.
//
// The protocol is length-free because every frame's size follows from n:
//
//   request   u32 magic 'B','R','D','6'
//             u32 n                        sequences in this batch
//             i32 ids[n * kSeqLen]         windows, row-major, right-padded
//             i32 length[n]                real ids in each row, 1..kSeqLen
//             f32 temperature[n]
//             u64 seed[n]
//
//   response  u32 status                   0 ok, 1 error
//     ok      i32 next[n]
//             u64 build_ns                 filling the (n, 256) tensor
//             u64 forward_ns               the model's own kernels
//             u64 copy_ns                  pulling the result off the device
//             u64 sample_ns                softmax and inverse-CDF, per row
//             u64 kernels                  CUDA kernels the forward launched
//             u64 to_device                 host->device copies (PCIe crossings)
//             u64 to_host                   device->host copies
//             f32 logits[n * vocab]        only with BRAID_EMIT_LOGITS=1
//     error   u32 length, then that many bytes of message
//
// The timings exist so that the server can subtract them from the wall time it
// measured and be left with the cost of the pipe itself, rather than quoting a
// round number for it. The kernel count is there because the engine falls back
// to the CPU for work below a size threshold, silently and by design: without
// it, a step that never touched the card is indistinguishable from one that did,
// and a forward time that halves as the batch grows has no explanation.
//
// They are also why the magic carries a version. A worker built against an older
// frame would answer a current server with something that parses as garbage
// rather than failing, and the version makes that a clean error instead.
//
// The lengths are why the padding can be on the right. The causal mask hides
// the future and nothing else, so with padding at the front the row being
// sampled attends to every pad id before it -- and id 0 in this alphabet is a
// tab, which is how an eleven-character prompt used to reach the model as two
// hundred and forty-five tabs. Padding on the right is unreachable from
// position length-1, but only if the caller says where that is, which is what
// this field is for and why the version went up rather than the field being
// inferred.
//
// The logits block is a diagnostic and is off unless BRAID_EMIT_LOGITS is set,
// which is why it is an environment variable rather than a protocol version.
// Serving does not want it: it is n * vocab floats a step that nobody reads, and
// the sampled id is the whole answer. What wants it is the batch-invariance
// measurement, which compares the *token* two runs produced and so can only see
// a difference in the arithmetic when it happens to cross a boundary of the
// sampler's inverse CDF. Comparing the logits measures the noise instead of the
// probability that the noise mattered.
//
// A zero-length read on stdin is a clean shutdown, not a failure: it is what
// the server closing the pipe looks like.

#include "charmodel.hpp"

#include "engine/autograd.hpp"
#include "engine/cuda.hpp"
#include "engine/data.hpp"
#include "engine/nn.hpp"
#include "engine/serialize.hpp"
#include "engine/tensor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace {

constexpr std::uint32_t kMagic = 0x36445242;  // 'BRD6' little-endian
constexpr std::uint32_t kStatusOK = 0;
constexpr std::uint32_t kStatusError = 1;

// Reads exactly n bytes, or returns false at a clean end of stream.
bool read_exact(void* dst, std::size_t n) {
    auto* out = static_cast<char*>(dst);
    std::size_t got = 0;
    while (got < n) {
        std::cin.read(out + got, static_cast<std::streamsize>(n - got));
        const std::streamsize just = std::cin.gcount();
        if (just <= 0) return false;
        got += static_cast<std::size_t>(just);
    }
    return true;
}

void write_all(const void* src, std::size_t n) {
    std::cout.write(static_cast<const char*>(src), static_cast<std::streamsize>(n));
}

void write_error(const std::string& message) {
    const std::uint32_t status = kStatusError;
    const auto length = static_cast<std::uint32_t>(message.size());
    write_all(&status, sizeof status);
    write_all(&length, sizeof length);
    write_all(message.data(), message.size());
    std::cout.flush();
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

// Samples one row of logits at a temperature, seeded only by that row's own
// seed. Nothing here may depend on n or on any other row: two sequences that
// share a batch must sample exactly as they would have alone, or the whole
// premise of batching them is false.
std::int32_t sample_row(const float* row, std::size_t vocab, float temperature,
                        std::uint64_t seed) {
    float best = row[0];
    for (std::size_t i = 1; i < vocab; ++i) best = std::max(best, row[i]);

    std::vector<float> probability(vocab);
    float total = 0.0f;
    for (std::size_t i = 0; i < vocab; ++i) {
        probability[i] = std::exp((row[i] - best) / temperature);
        total += probability[i];
    }

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> pick(0.0f, total);
    float threshold = pick(rng);
    for (std::size_t i = 0; i < vocab; ++i) {
        threshold -= probability[i];
        if (threshold <= 0.0f) return static_cast<std::int32_t>(i);
    }
    return static_cast<std::int32_t>(vocab - 1);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: braid_worker <checkpoint-prefix>\n";
        return 2;
    }
    const std::string prefix = argv[1];

#ifdef _WIN32
    // Without this the runtime translates every 0x0A in the stream into 0x0D
    // 0x0A on the way out and back on the way in, which corrupts binary frames
    // in a way that looks like a protocol bug for as long as you let it.
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif

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

    // A diagnostic hook, not a serving knob. Forcing one matmul variant is how
    // the discontinuity at a batch of six was traced: if pinning the kernel
    // flattens it, the cause is which kernel Auto picks and not how much work
    // there is. Auto is what braid actually serves with.
    // A diagnostic, read once. See the note on the logits block above.
    const bool emit_logits = std::getenv("BRAID_EMIT_LOGITS") != nullptr;
    if (emit_logits) {
        std::cerr << "braid_worker: emitting logits, which serving does not want" << std::endl;
    }

    if (const char* forced = std::getenv("BRAID_MATMUL_KERNEL")) {
        const std::string want = forced;
        engine::cuda::MatmulKernel kernel = engine::cuda::MatmulKernel::Auto;
        if (want == "naive") {
            kernel = engine::cuda::MatmulKernel::Naive;
        } else if (want == "tiled") {
            kernel = engine::cuda::MatmulKernel::Tiled;
        } else if (want == "register") {
            kernel = engine::cuda::MatmulKernel::RegisterTiled;
        } else if (want == "vectorized") {
            kernel = engine::cuda::MatmulKernel::Vectorized;
        } else if (want != "auto") {
            std::cerr << "BRAID_MATMUL_KERNEL=" << want << " is not a kernel name\n";
            return 2;
        }
        engine::cuda::set_matmul_kernel(kernel);
    }

    // Announced on stderr, which the server logs. available() says whether the
    // binary has a usable device at all; the kernel counter at shutdown says
    // whether it was ever actually used, because the engine falls back to the
    // CPU silently and a GPU that was never touched should not be reported as
    // one that was.

    const auto device = engine::cuda::device_info();
    std::cerr << "braid_worker ready: vocab " << vocab.size() << ", cuda "
              << (engine::cuda::available() ? "yes" : "no");
    if (device) std::cerr << " (" << device->name << ")";
    // The thresholds decide whether an operation goes to the card at all, and
    // at this model's size they decide it differently for a batch of one than
    // for a batch of eight. A run that does not say which ones it used cannot
    // be compared against another run, so every run says.
    std::cerr << ", min_matmul_flops " << engine::cuda::min_matmul_flops()
              << ", min_elements " << engine::cuda::min_elementwise_elements() << "\n";
    std::cerr.flush();

    // One mask per width actually used, built on demand and kept. A run of a
    // sequence walks its length upward one position at a time, so this fills
    // with the widths that run touches and then stops allocating. Rebuilding a
    // (S, S) mask every step would be host work and a PCIe upload per step, for
    // a tensor that never changes.
    std::map<std::size_t, engine::Tensor> masks;
    const auto mask_for = [&masks](std::size_t width) -> const engine::Tensor& {
        auto found = masks.find(width);
        if (found == masks.end()) {
            found = masks.emplace(width, engine::nn::causal_mask(width)).first;
        }
        return found->second;
    };

    engine::autograd::NoGradGuard no_grad;

    std::vector<std::int32_t> windows;
    std::vector<std::int32_t> lengths;
    std::vector<float> temperatures;
    std::vector<std::uint64_t> seeds;
    std::vector<std::int32_t> next;
    std::vector<float> sampled;  // only filled when BRAID_EMIT_LOGITS is set

    for (;;) {
        std::uint32_t magic = 0;
        if (!read_exact(&magic, sizeof magic)) break;  // the server closed the pipe
        if (magic != kMagic) {
            write_error("bad frame magic");
            return 1;
        }

        std::uint32_t n = 0;
        if (!read_exact(&n, sizeof n)) break;
        if (n == 0) {
            write_error("a step with no sequences in it");
            return 1;
        }

        const std::size_t ids_count = static_cast<std::size_t>(n) * braid::kSeqLen;
        windows.resize(ids_count);
        lengths.resize(n);
        temperatures.resize(n);
        seeds.resize(n);
        if (!read_exact(windows.data(), ids_count * sizeof(std::int32_t)) ||
            !read_exact(lengths.data(), n * sizeof(std::int32_t)) ||
            !read_exact(temperatures.data(), n * sizeof(float)) ||
            !read_exact(seeds.data(), n * sizeof(std::uint64_t))) {
            std::cerr << "truncated frame\n";
            return 1;
        }

        // Checked rather than clamped: a length outside the window means the
        // two sides disagree about the frame, and sampling some other row would
        // answer the wrong question convincingly.
        for (std::uint32_t i = 0; i < n; ++i) {
            if (lengths[i] < 1 || static_cast<std::size_t>(lengths[i]) > braid::kSeqLen) {
                write_error("a row claims a length outside the window");
                return 1;
            }
        }

        // The longest row decides the width of the step. Everything past it in
        // every row is padding whose logits nobody reads, and computing it was
        // costing the whole batch: at a batch of 32 a step over 256 positions is
        // 78 ms and one over 32 is 11.
        //
        // A shorter row is still correct at this width. Its own padding sits
        // between its length and the width, and the causal mask at the position
        // being sampled -- length-1 -- reaches only 0..length-1, which is all
        // real. That is why the width can be the maximum rather than a value
        // every row has to agree on, and it is the second thing the padding
        // moving to the right bought.
        std::size_t width = 1;
        for (std::uint32_t i = 0; i < n; ++i) {
            width = std::max(width, static_cast<std::size_t>(lengths[i]));
        }

        try {
            using clock = std::chrono::steady_clock;
            const auto t0 = clock::now();

            engine::Tensor ids({n, width}, 0.0f, false);
            float* id_values = ids.data();
            for (std::uint32_t i = 0; i < n; ++i) {
                const std::int32_t* row = windows.data() + static_cast<std::size_t>(i) * braid::kSeqLen;
                float* out = id_values + static_cast<std::size_t>(i) * width;
                for (std::size_t j = 0; j < width; ++j) out[j] = static_cast<float>(row[j]);
            }
            const auto t1 = clock::now();
            const std::size_t kernels_before = engine::cuda::kernels_launched();
            // An operation the engine keeps on the host still has to get its
            // inputs there and its result back, so a step that launches fewer
            // kernels crosses PCIe more often. Counting both says which of the
            // two a change in kernel count actually was.
            const auto transfers_before = engine::cuda::transfer_stats();

            // The one forward pass. n rows in, n rows of logits out, and the
            // model has no way of knowing the rows belong to different callers.
            const engine::Tensor logits = model.forward(ids, mask_for(width));  // (n, W, vocab)

            // The kernels are asynchronous, so forward() returning says only
            // that they were launched. Synchronising here ends the compute and
            // starts the copy, which is the only way to tell the two apart --
            // and it costs nothing, because data() below would synchronise
            // anyway.
            engine::cuda::synchronize();
            const auto t_compute = clock::now();

            // This pulls (n, width, vocab) to the host when the sampling below
            // reads one row of it per sequence: the last real position of each.
            // The other width-1 come across for nothing, every step. Measured
            // rather than assumed, because a device-side slice is an engine
            // change and "it must be slow" is not a reason to make one.
            const float* data = logits.data();
            const auto t2 = clock::now();

            const std::size_t vocab_size = vocab.size();
            next.resize(n);
            if (emit_logits) sampled.resize(static_cast<std::size_t>(n) * vocab_size);
            for (std::uint32_t i = 0; i < n; ++i) {
                // The last *real* position of this row, not the last position
                // of the window. Everything past it is padding, which the
                // causal mask makes unreachable from here.
                const std::size_t last = static_cast<std::size_t>(lengths[i]) - 1;
                const float* row = data + (static_cast<std::size_t>(i) * width + last) * vocab_size;
                if (emit_logits) {
                    std::copy_n(row, vocab_size, sampled.begin() + static_cast<std::size_t>(i) * vocab_size);
                }
                next[i] = sample_row(row, vocab_size, temperatures[i], seeds[i]);
            }
            const auto t3 = clock::now();

            const auto ns = [](clock::time_point from, clock::time_point to) {
                return static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(to - from).count());
            };
            const std::uint64_t build_ns = ns(t0, t1);
            const std::uint64_t forward_ns = ns(t1, t_compute);
            const std::uint64_t copy_ns = ns(t_compute, t2);
            const std::uint64_t sample_ns = ns(t2, t3);
            const auto kernels =
                static_cast<std::uint64_t>(engine::cuda::kernels_launched() - kernels_before);
            const auto transfers_after = engine::cuda::transfer_stats();
            const auto to_device = static_cast<std::uint64_t>(transfers_after.to_device_count -
                                                              transfers_before.to_device_count);
            const auto to_host = static_cast<std::uint64_t>(transfers_after.to_host_count -
                                                            transfers_before.to_host_count);

            const std::uint32_t status = kStatusOK;
            write_all(&status, sizeof status);
            write_all(next.data(), static_cast<std::size_t>(n) * sizeof(std::int32_t));
            write_all(&build_ns, sizeof build_ns);
            write_all(&forward_ns, sizeof forward_ns);
            write_all(&copy_ns, sizeof copy_ns);
            write_all(&sample_ns, sizeof sample_ns);
            write_all(&kernels, sizeof kernels);
            write_all(&to_device, sizeof to_device);
            write_all(&to_host, sizeof to_host);
            if (emit_logits) {
                write_all(sampled.data(), sampled.size() * sizeof(float));
            }
            std::cout.flush();
        } catch (const std::exception& e) {
            write_error(e.what());
            return 1;
        }
    }

    std::cerr << "braid_worker stopping, kernels launched " << engine::cuda::kernels_launched()
              << ", failed " << engine::cuda::kernels_failed() << "\n";
    return 0;
}
