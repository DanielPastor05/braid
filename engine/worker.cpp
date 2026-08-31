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
//   request   u32 magic 'B','R','D','7'
//             u32 n                        sequences in this batch
//             i32 ids[n * kSeqLen]         windows, row-major, right-padded
//             i32 length[n]                real ids in each row, 1..kSeqLen
//             i32 slot[n]                  cache slot, or -1 for none
//             f32 temperature[n]
//             u64 seed[n]
//
// The window is sent in full even when a slot is given, and that is deliberate
// rather than left over. The cache is an optimisation the worker may or may not
// have -- a worker that has just started after a failover has none -- and a frame
// carrying the whole history means it can always fall back to recomputing. The
// scheduler stays the only authority on what a sequence has said, so failover is
// still a retry of the same bytes on another worker, which is the property this
// whole design rests on.
//
// A slot of -1 means do not cache this row. A slot whose cache has not reached
// length-1 is refilled from the window first, in one forward pass rather than one
// per position: that covers a sequence arriving with a prompt, a slot being
// handed to somebody new, and a worker that has never seen this sequence.
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

constexpr std::uint32_t kMagic = 0x37445242;  // 'BRD7' little-endian
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

    // The key/value cache, off unless BRAID_CACHE is set, because whether it is
    // worth having is a measured question with a mixed answer.
    //
    // braid_bench_decode sweeps it. With the capacity rounded to a block of the
    // history rather than to the whole context, a step is 2.4x at a batch of
    // sixteen and twenty-nine positions of history, 4.1x at thirty-two, and 9.6x
    // at four hundred and fifty. At a batch of **one** it is 0.9x -- slower --
    // because a step that short is already at the ~2.4 ms floor of 177 kernel
    // launches, and a cache makes kernels smaller rather than fewer while adding
    // a gather to pay for. So a batch of one recomputes even when the cache is
    // on.
    //
    // Off by default until the server has been measured both ways end to end. A
    // change that costs the single-client number to buy the loaded one is a
    // trade, and this repository publishes trades rather than the flattering
    // half of them.
    const bool cache_enabled = std::getenv("BRAID_CACHE") != nullptr;

    // Blocks of sixteen positions: internal/kvmem's default, chosen there for
    // under six percent internal fragmentation. Rounding the step's width up to
    // one of these instead of up to the context is the whole reason the cache
    // pays -- an over-allocated cache is measurably slower than no cache.
    constexpr std::size_t kBlock = 16;

    // One pool per block, each (max_slots, heads, kSeqLen, head_dim), allocated
    // on the first frame that names a slot because the server decides how many
    // there are. `filled` in each says how far every slot has been written.
    std::vector<engine::nn::KVCache> pool;
    std::size_t pool_slots = 0;
    const auto ensure_pool = [&](std::size_t slots) {
        if (pool_slots >= slots) return;
        // Growing means starting over: the slots are being renumbered by the
        // server, so nothing in the old pool is addressable any more. It happens
        // once, on the first frame.
        pool = model.make_caches(slots, braid::kSeqLen);

        // Put it on the card, once, and this is not optional.
        //
        // Every device path in the engine is admitted on residency, and both of
        // the ones this pool needs -- select_rows_window to read it and
        // scatter_rows to write it -- *require* the pool to be resident and
        // silently take the host path when it is not. A pool built by
        // make_caches has only ever existed on the host, so nothing would ever
        // put it there: each step would gather on the host, upload, step,
        // download, and scatter on the host. The cache would live on the wrong
        // side of PCIe for its whole life.
        //
        // Adding zeros to zeros is the cheapest operation that leaves the result
        // resident. It looks like a no-op and it is the opposite of one.
        for (auto& block : pool) {
            block.keys = block.keys + block.keys;
            block.values = block.values + block.values;
        }
        engine::cuda::synchronize();

        pool_slots = slots;
    };

    // Allocated at start rather than on the first frame that names a slot, when
    // the server says how many there will be.
    //
    // The pool is (slots, heads, kSeqLen, head_dim) twice over for every block:
    // 1.2 GB at sixty-four slots, and another 1.2 transiently while it is being
    // put on the card. Doing that lazily charges the whole thing to whichever
    // request happens to arrive first, and it showed up as a TTFT p95 of 194 ms
    // against a step of six -- a cold start wearing the costume of a latency
    // regression.
    if (cache_enabled) {
        const char* slots_env = std::getenv("BRAID_CACHE_SLOTS");
        const long wanted = slots_env ? std::strtol(slots_env, nullptr, 10) : 0;
        if (wanted > 0) ensure_pool(static_cast<std::size_t>(wanted));
    }

    std::vector<std::int32_t> windows;
    std::vector<std::int32_t> lengths;
    std::vector<std::int32_t> slots;
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
        slots.resize(n);
        temperatures.resize(n);
        seeds.resize(n);
        if (!read_exact(windows.data(), ids_count * sizeof(std::int32_t)) ||
            !read_exact(lengths.data(), n * sizeof(std::int32_t)) ||
            !read_exact(slots.data(), n * sizeof(std::int32_t)) ||
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

            // Cached only above a batch of one. At a batch of one the step is
            // already launch-bound and the cache measures 0.9x -- see the note
            // where cache_enabled is read.
            bool cached = cache_enabled && n > 1;
            if (cached) {
                for (std::uint32_t i = 0; i < n; ++i) {
                    if (slots[i] < 0) cached = false;
                }
            }

            // Positions per row in `logits`, and which of them a row samples.
            // The uncached path returns the whole window and samples at
            // length-1; the cached one returns the single new position.
            std::size_t pitch = width;
            bool sample_at_length = true;

            engine::Tensor logits;
            if (!cached) {
                // The one forward pass. n rows in, n rows of logits out, and the
                // model has no way of knowing the rows belong to different
                // callers.
                logits = model.forward(ids, mask_for(width));  // (n, W, vocab)
            } else {
                std::size_t highest = 0;
                for (std::uint32_t i = 0; i < n; ++i) {
                    highest = std::max(highest, static_cast<std::size_t>(slots[i]));
                }
                ensure_pool(highest + 1);

                // The width of the step, rounded up to a block. Rounding to the
                // context instead is what makes a cache slower than no cache.
                const std::size_t cap =
                    std::min(braid::kSeqLen, ((width + kBlock - 1) / kBlock) * kBlock);

                // Any row whose slot has not reached length-1 is refilled
                // from the window: a sequence arriving with a prompt, a slot
                // handed to somebody new, or this worker having never seen it.
                //
                // All of them in **one** forward pass, not one each. Rows that
                // arrive together otherwise serialise their prefills inside the
                // step, and every other row in the batch waits behind them --
                // which is what a TTFT p50 of 66 ms against a 6 ms step looks
                // like from outside.
                //
                // They are padded to the longest prompt among them, and the
                // padding is safe for the same reason the window's is: each row
                // writes `take` positions into its cache, but `filled` is then
                // set to its own real length and every later mask reaches only
                // that far. What the padding computed is unreachable.
                std::vector<std::size_t> refill;      // rows of this batch
                std::vector<std::size_t> refill_slot;
                std::vector<std::size_t> refill_have;
                std::size_t longest = 0;
                for (std::uint32_t i = 0; i < n; ++i) {
                    const std::size_t slot = static_cast<std::size_t>(slots[i]);
                    const std::size_t have = static_cast<std::size_t>(lengths[i]) - 1;
                    if (pool.front().filled[slot] == have) continue;

                    for (auto& block : pool) block.filled[slot] = 0;
                    if (have == 0) continue;

                    refill.push_back(i);
                    refill_slot.push_back(slot);
                    refill_have.push_back(have);
                    longest = std::max(longest, have);
                }

                if (!refill.empty()) {
                    const std::size_t k = refill.size();
                    engine::Tensor prompts({k, longest}, 0.0f, false);
                    for (std::size_t r = 0; r < k; ++r) {
                        const std::int32_t* row =
                            windows.data() + refill[r] * braid::kSeqLen;
                        float* out = prompts.data() + r * longest;
                        for (std::size_t j = 0; j < refill_have[r]; ++j) {
                            out[j] = static_cast<float>(row[j]);
                        }
                    }
                    // Rounded to a block, not to the context. A cached forward
                    // attends over its whole capacity, so prefilling into a
                    // 1024-wide cache is a 1024-wide step to write forty
                    // positions -- the same mistake this file's capacity note is
                    // about, made where the note was not looking. It cost the
                    // server half its throughput.
                    const std::size_t room =
                        std::min(braid::kSeqLen, ((longest + kBlock - 1) / kBlock) * kBlock);
                    auto fresh = model.make_caches(k, room);
                    const std::vector<std::size_t> from_zero(k, 0);
                    (void)model.forward_cached(prompts, from_zero, fresh);

                    for (std::size_t b = 0; b < braid::kBlocks; ++b) {
                        // One scatter for the whole set, at `longest` positions
                        // each: the tails past a row's own length are unread.
                        pool[b].keys.scatter_rows(fresh[b].keys.slice(2, 0, longest), 2,
                                                  refill_slot, from_zero);
                        pool[b].values.scatter_rows(fresh[b].values.slice(2, 0, longest), 2,
                                                    refill_slot, from_zero);
                        for (std::size_t r = 0; r < k; ++r) {
                            pool[b].filled[refill_slot[r]] = refill_have[r];
                        }
                    }
                }

                std::vector<std::size_t> active(n), at(n), zero(n, 0);
                for (std::uint32_t i = 0; i < n; ++i) {
                    active[i] = static_cast<std::size_t>(slots[i]);
                    at[i] = static_cast<std::size_t>(lengths[i]) - 1;
                }

                // The active slots at the step's width, in one operation.
                //
                // Composing gather and slice moves the intersection twice over:
                // gathering first takes `n` rows at the *full context*,
                // narrowing first takes *every slot* at the right width, and
                // what is wanted is `n * cap`. Serving sixteen rows at width
                // forty-eight from sixty-four slots of a thousand positions, the
                // better composition still moves four times too much and the
                // natural one twenty-one times too much -- which is why this
                // server was slower with its cache than without it until
                // select_rows_window existed.
                std::vector<engine::nn::KVCache> compact;
                compact.reserve(braid::kBlocks);
                for (std::size_t b = 0; b < braid::kBlocks; ++b) {
                    engine::nn::KVCache one(1, braid::kHeads, 1, braid::kDModel / braid::kHeads);
                    one.keys = pool[b].keys.select_rows_window(active, 2, 0, cap);
                    one.values = pool[b].values.select_rows_window(active, 2, 0, cap);
                    one.filled.assign(at.begin(), at.end());
                    compact.push_back(std::move(one));
                }

                // One new id per row: the position each is actually asking about.
                engine::Tensor fresh({n, 1}, 0.0f, false);
                for (std::uint32_t i = 0; i < n; ++i) {
                    const std::int32_t* row =
                        windows.data() + static_cast<std::size_t>(i) * braid::kSeqLen;
                    fresh.data()[i] = static_cast<float>(row[at[i]]);
                }

                logits = model.forward_cached(fresh, at, compact);  // (n, 1, vocab)
                pitch = 1;
                sample_at_length = false;

                // And back to the slots they came from. The whole compact cache
                // rather than each row's one new position: extracting a
                // different column per row would need another gather, and at
                // this width the extra bytes are a tenth of a millisecond.
                for (std::size_t b = 0; b < braid::kBlocks; ++b) {
                    pool[b].keys.scatter_rows(compact[b].keys, 2, active, zero);
                    pool[b].values.scatter_rows(compact[b].values, 2, active, zero);
                    for (std::uint32_t i = 0; i < n; ++i) {
                        pool[b].filled[active[i]] = at[i] + 1;
                    }
                }
            }

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
                const std::size_t last = sample_at_length
                                             ? static_cast<std::size_t>(lengths[i]) - 1
                                             : 0;
                const float* row = data + (static_cast<std::size_t>(i) * pitch + last) * vocab_size;
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
