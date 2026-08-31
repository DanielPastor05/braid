// Does decoding through the cache give the same answer as recomputing?
//
// This is the only property that matters about a key/value cache, and it is the
// one that is easy to lose: a cache that is subtly wrong does not crash, it
// produces fluent text that differs from what the model would have said. The
// check is therefore against the uncached forward over the same ids, not
// against a stored expectation.
//
// Three arrangements, because the interesting failures are about rows being at
// different places:
//
//   1. One row, decoded a position at a time from empty.
//   2. A batch whose rows are all at the same position -- the arrangement a
//      shared mask would also handle, so this passing proves little on its own.
//   3. A batch whose rows are at *different* positions, which is what continuous
//      batching produces and what the per-row mask and per-row positional
//      gather exist for. This is the one that fails if either is wrong.
//
//   braid_test_cached <model-prefix>

#include "charmodel.hpp"

#include "engine/autograd.hpp"
#include "engine/data.hpp"
#include "engine/serialize.hpp"
#include "engine/transformer.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::cout << (ok ? "  ok   " : "  FAIL ") << what << "\n";
    if (!ok) ++failures;
}

// The largest absolute difference between the logits of the row's last position.
double worst_gap(const engine::Tensor& a, const engine::Tensor& b, std::size_t vocab) {
    double worst = 0.0;
    for (std::size_t i = 0; i < vocab; ++i) {
        worst = std::max(worst, static_cast<double>(std::abs(a.data()[i] - b.data()[i])));
    }
    return worst;
}

// One row of `ids`, recomputed from scratch over its first `len` positions, with
// the last position's logits returned.
std::vector<float> uncached_last(braid::CharModel& model, const std::vector<std::int32_t>& ids,
                                 std::size_t len, std::size_t vocab) {
    engine::Tensor window({1, len}, 0.0f, false);
    for (std::size_t i = 0; i < len; ++i) window.data()[i] = static_cast<float>(ids[i]);
    const engine::Tensor mask = engine::nn::causal_mask(len);
    const engine::Tensor logits = model.forward(window, mask);
    const float* row = logits.data() + (len - 1) * vocab;
    return std::vector<float>(row, row + vocab);
}

}  // namespace

// Everything below runs inside a try, because the engine reports misuse by
// throwing and an uncaught exception on Windows is a bare 0xC0000409 with no
// message at all. Writing this test cost ten minutes to a call that had already
// been rejected, correctly and with an explanation, that nothing printed.
int run(int argc, char** argv);

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "threw: " << e.what() << std::endl;
        return 1;
    }
}

int run(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: braid_test_cached <model-prefix>\n";
        return 2;
    }
    const std::string prefix = argv[1];

    std::string alphabet;
    {
        std::ifstream in(prefix + ".vocab", std::ios::binary);
        if (!in) {
            std::cerr << "cannot read " << prefix << ".vocab\n";
            return 1;
        }
        alphabet.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    const engine::data::CharVocab vocab(alphabet);
    braid::CharModel model(vocab.size());
    {
        auto named = model.named_parameters();
        engine::load_parameters(named, prefix + ".bin");
    }
    model.train(false);
    engine::autograd::NoGradGuard no_grad;

    const std::size_t V = vocab.size();
    std::mt19937 rng(20260831);
    std::uniform_int_distribution<int> pick(0, static_cast<int>(V) - 1);

    // A cache narrower than the model's context: nothing here needs 1024, and a
    // capacity that is not kSeqLen is worth exercising because the mask is built
    // against the capacity rather than the context.
    constexpr std::size_t kCapacity = 96;

    std::cout << "cached decode against recomputation, " << V << " symbols, capacity "
              << kCapacity << "\n";

    // ---- 1. one row, from empty, a position at a time -------------------
    {
        std::vector<std::int32_t> ids(24);
        for (auto& id : ids) id = pick(rng);

        auto caches = model.make_caches(1, kCapacity);
        double worst = 0.0;
        for (std::size_t t = 0; t < ids.size(); ++t) {
            engine::Tensor step({1, 1}, static_cast<float>(ids[t]), false);
            const engine::Tensor got = model.forward_cached(step, {t}, caches);
            const std::vector<float> want = uncached_last(model, ids, t + 1, V);

            engine::Tensor want_t({V}, 0.0f, false);
            std::copy(want.begin(), want.end(), want_t.data());
            worst = std::max(worst, worst_gap(got, want_t, V));
        }
        std::printf("  worst logit gap over 24 positions: %.3e\n", worst);
        check(worst < 1e-2, "one row decoded a position at a time matches recomputation");
    }

    // ---- 1b. many new positions at once, which is a prefill --------------
    //
    // A sequence arriving with a prompt has to get that prompt into the cache,
    // and doing it a position at a time is one forward pass per character. One
    // call with `take` new positions is the whole point -- but it only works if
    // a new position may attend to the new positions before it in the same
    // call, which is a property of the mask this file builds rather than of the
    // engine. Untested, it is exactly the kind of thing that produces fluent
    // wrong text.
    {
        std::vector<std::int32_t> ids(18);
        for (auto& id : ids) id = pick(rng);

        auto caches = model.make_caches(1, kCapacity);
        engine::Tensor prompt({1, 12}, 0.0f, false);
        for (std::size_t i = 0; i < 12; ++i) prompt.data()[i] = static_cast<float>(ids[i]);

        // Twelve positions in one call, starting from an empty cache.
        const engine::Tensor got = model.forward_cached(prompt, {0}, caches);

        // Every one of them should match what recomputing that prefix gives.
        double worst = 0.0;
        for (std::size_t t = 0; t < 12; ++t) {
            const std::vector<float> want = uncached_last(model, ids, t + 1, V);
            engine::Tensor want_t({V}, 0.0f, false);
            std::copy(want.begin(), want.end(), want_t.data());
            engine::Tensor got_t({V}, 0.0f, false);
            std::copy_n(got.data() + t * V, V, got_t.data());
            worst = std::max(worst, worst_gap(got_t, want_t, V));
        }
        std::printf("  worst logit gap, 12 positions in one call: %.3e\n", worst);
        check(worst < 1e-2, "a multi-position prefill matches recomputation at every position");

        // And decoding on afterwards has to continue from where the prefill
        // left the cache, not from where a fresh one would.
        double after = 0.0;
        for (std::size_t t = 12; t < ids.size(); ++t) {
            engine::Tensor step({1, 1}, static_cast<float>(ids[t]), false);
            const engine::Tensor one = model.forward_cached(step, {t}, caches);
            const std::vector<float> want = uncached_last(model, ids, t + 1, V);
            engine::Tensor want_t({V}, 0.0f, false);
            std::copy(want.begin(), want.end(), want_t.data());
            after = std::max(after, worst_gap(one, want_t, V));
        }
        std::printf("  worst logit gap, decoding on from a prefill: %.3e\n", after);
        check(after < 1e-2, "and decoding continues correctly from a prefilled cache");
    }

    // ---- 2. a batch with every row at the same position ------------------
    {
        constexpr std::size_t kRows = 4;
        std::vector<std::vector<std::int32_t>> ids(kRows, std::vector<std::int32_t>(12));
        for (auto& row : ids)
            for (auto& id : row) id = pick(rng);

        auto caches = model.make_caches(kRows, kCapacity);
        double worst = 0.0;
        for (std::size_t t = 0; t < 12; ++t) {
            engine::Tensor step({kRows, 1}, 0.0f, false);
            for (std::size_t r = 0; r < kRows; ++r) step.data()[r] = static_cast<float>(ids[r][t]);

            const std::vector<std::size_t> at(kRows, t);
            const engine::Tensor got = model.forward_cached(step, at, caches);
            for (std::size_t r = 0; r < kRows; ++r) {
                const std::vector<float> want = uncached_last(model, ids[r], t + 1, V);
                engine::Tensor want_t({V}, 0.0f, false);
                std::copy(want.begin(), want.end(), want_t.data());
                engine::Tensor got_r({V}, 0.0f, false);
                std::copy_n(got.data() + r * V, V, got_r.data());
                worst = std::max(worst, worst_gap(got_r, want_t, V));
            }
        }
        std::printf("  worst logit gap, 4 aligned rows: %.3e\n", worst);
        check(worst < 1e-2, "an aligned batch matches recomputation");
    }

    // ---- 3. a batch whose rows are at different positions ----------------
    //
    // The one that matters. Each row starts at its own offset and advances at
    // the same rate, so no two rows share a position at any step -- which is
    // what a shared mask or a sliced positional table would get wrong.
    {
        constexpr std::size_t kRows = 5;
        const std::size_t start[kRows] = {0, 3, 9, 17, 30};
        std::vector<std::vector<std::int32_t>> ids(kRows);
        for (std::size_t r = 0; r < kRows; ++r) {
            ids[r].resize(start[r] + 10);
            for (auto& id : ids[r]) id = pick(rng);
        }

        // Rows cannot be prefilled to different depths in one batched call --
        // every row of a call advances by the same number of positions. So each
        // is prefilled alone and its cache copied into the batched one, which is
        // the same scatter the worker will do with copy_into_rows.
        std::vector<std::size_t> at(kRows);
        for (std::size_t r = 0; r < kRows; ++r) at[r] = start[r];

        auto caches = model.make_caches(kRows, kCapacity);
        for (std::size_t r = 0; r < kRows; ++r) {
            auto one = model.make_caches(1, kCapacity);
            for (std::size_t t = 0; t < start[r]; ++t) {
                engine::Tensor step({1, 1}, static_cast<float>(ids[r][t]), false);
                (void)model.forward_cached(step, {t}, one);
            }
            for (std::size_t b = 0; b < braid::kBlocks; ++b) {
                // copy_into, not copy_into_rows: this is one whole row of the
                // batch axis going into slot r, which is a single start along
                // axis 0. copy_into_rows takes a start *per row* along a later
                // axis -- it is for advancing every row's own write position,
                // which is what the worker does per step and not what a scatter
                // of one row into a slot is.
                caches[b].keys.copy_into(one[b].keys, 0, r);
                caches[b].values.copy_into(one[b].values, 0, r);
                caches[b].filled[r] = one[b].filled[0];
            }
        }

        double worst = 0.0;
        for (std::size_t k = 0; k < 10; ++k) {
            engine::Tensor step({kRows, 1}, 0.0f, false);
            for (std::size_t r = 0; r < kRows; ++r) {
                step.data()[r] = static_cast<float>(ids[r][at[r]]);
            }
            const engine::Tensor got = model.forward_cached(step, at, caches);

            for (std::size_t r = 0; r < kRows; ++r) {
                const std::vector<float> want = uncached_last(model, ids[r], at[r] + 1, V);
                engine::Tensor want_t({V}, 0.0f, false);
                std::copy(want.begin(), want.end(), want_t.data());
                engine::Tensor got_r({V}, 0.0f, false);
                std::copy_n(got.data() + r * V, V, got_r.data());
                worst = std::max(worst, worst_gap(got_r, want_t, V));
            }
            for (std::size_t r = 0; r < kRows; ++r) ++at[r];
        }
        std::printf("  worst logit gap, 5 rows at different positions: %.3e\n", worst);
        check(worst < 1e-2, "a misaligned batch matches recomputation");
    }

    std::cout << (failures == 0 ? "\nall cached decodes agree with recomputation\n"
                                : "\nsomething disagrees\n");
    return failures == 0 ? 0 : 1;
}
