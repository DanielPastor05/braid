// The model braid serves: the character-level Transformer from cpp-ai-engine's
// charlm_demo, lifted out of that demo's anonymous namespace so that the
// trainer and the server can both build the identical thing.
//
// The geometry is copied deliberately rather than parameterised. A checkpoint
// written by one binary and read by another is only safe if both agree on every
// dimension, and engine::load_parameters checks shapes on load -- so a mismatch
// here is a refused file rather than a model that quietly reads the weights
// wrong. If these constants ever change, old checkpoints stop loading, which is
// the correct outcome.
#ifndef BRAID_CHARMODEL_HPP
#define BRAID_CHARMODEL_HPP

#include "engine/nn.hpp"
#include "engine/tensor.hpp"
#include "engine/transformer.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace braid {

// The geometry, chosen so that the GPU is the thing being measured.
//
// The first version of this model was 172 728 parameters over a 64-id context,
// and every serving number taken from it was really a number about launch
// overhead: the forward went from 0.77 ms at a batch of one to 2.32 ms at
// thirty-two, which is a fixed cost amortising rather than arithmetic scaling.
// Nothing measured there transferred to anything larger.
//
// At these sizes a step is roughly 20 MFLOP per token over 256 positions and
// thirty-two sequences -- about 164 GFLOP -- against a 3060 Ti's ~16 TFLOPS.
// That is tens of milliseconds of real work, and the profile stops being a
// story about overhead.
constexpr std::size_t kSeqLen = 1024;
constexpr std::size_t kDModel = 384;
constexpr std::size_t kHeads = 6;  // 64 per head
constexpr std::size_t kFeedForward = 1536;
constexpr std::size_t kBlocks = 6;

// Assembled by hand rather than through nn::Sequential, because a Transformer
// block takes a mask as a second argument and Sequential::forward(input) has
// nowhere to put it.
struct CharModel {
    engine::nn::Embedding embedding;
    std::vector<engine::nn::TransformerBlock> blocks;
    engine::nn::Linear head;
    engine::Tensor positions;

    explicit CharModel(std::size_t vocab)
        : embedding(vocab, kDModel),
          head(kDModel, vocab),
          positions(engine::nn::positional_encoding(kSeqLen, kDModel)) {
        blocks.reserve(kBlocks);
        for (std::size_t i = 0; i < kBlocks; ++i) {
            blocks.emplace_back(kDModel, kHeads, kFeedForward);
        }
    }

    // (B, S) indices in, (B, S, vocab) logits out. B is whatever the caller
    // stacked: one sequence when training a batch, or n concurrent requests
    // when serving them together. The model cannot tell the difference, which
    // is the entire premise of batching them.
    // S may be narrower than kSeqLen. Serving hands in only as many positions as
    // the longest sequence in the batch actually has, because everything past it
    // is padding whose logits nobody reads; training hands in the full window.
    // The positional table is cut to match, and only when it has to be -- the
    // slice is a copy, and the training path should not pay for one.
    engine::Tensor forward(const engine::Tensor& ids, const engine::Tensor& mask) {
        const std::size_t take = ids.shape()[1];
        engine::Tensor h = embedding.forward(ids) +
                           (take == kSeqLen ? positions : positions.slice(0, 0, take));
        for (engine::nn::TransformerBlock& block : blocks) h = block.forward(h, &mask);
        return head.forward(h);
    }

    // The same model over a cache, for decoding.
    //
    // `ids` is (B, S) of the *new* positions only -- one per row while
    // generating, more on a prefill -- and `at` is where each row's first new
    // position sits in its own sequence. `caches` is one KVCache per block, all
    // (B, heads, capacity, head_dim), holding everything before them.
    //
    // Two things differ from the uncached path and both are about rows being at
    // different positions, which is what continuous batching guarantees:
    //
    // The positional encoding is gathered per row rather than sliced. A slice
    // takes the same rows of the table for the whole batch, which is only right
    // when every row is at the same position -- the arrangement this server
    // measured at 2% of steps.
    //
    // The mask is (B, heads, S, capacity), built from each row's own absolute
    // position. Attention runs over the full capacity because slicing the cache
    // would copy it every step, so the unwritten tail is zeros, and exp(0) is 1:
    // the mask is the only thing keeping positions that do not exist yet out of
    // the softmax.
    //
    // Inference only. The caches are written in place, so a backward through a
    // value the next step overwrites would describe a forward that never
    // happened; the engine throws rather than allow it.
    engine::Tensor forward_cached(const engine::Tensor& ids, const std::vector<std::size_t>& at,
                                  std::vector<engine::nn::KVCache>& caches) {
        const std::size_t batch = ids.shape()[0];
        const std::size_t take = ids.shape()[1];
        if (caches.size() != kBlocks) {
            throw std::invalid_argument("forward_cached needs one cache per block");
        }
        if (at.size() != batch) {
            throw std::invalid_argument("forward_cached needs one position per row");
        }
        const std::size_t capacity = caches.front().capacity();

        // Row r's new positions are at[r] .. at[r]+take-1, so its slice of the
        // positional table starts in a different place from its neighbours'.
        engine::Tensor pos({batch, take, kDModel}, 0.0f, false);
        {
            float* out = pos.data();
            const float* table = positions.data();
            for (std::size_t r = 0; r < batch; ++r) {
                if (at[r] + take > kSeqLen) {
                    throw std::out_of_range("a row would decode past the context");
                }
                std::copy_n(table + at[r] * kDModel, take * kDModel,
                            out + r * take * kDModel);
            }
        }

        engine::Tensor h = embedding.forward(ids) + pos;

        // One mask per row, repeated per head because the engine broadcasts by
        // suffix after dropping leading ones: a (B, 1, S, capacity) mask would
        // be rejected rather than expanded across the heads.
        // The mask is *additive*: it is added to the scores before the softmax,
        // so a position that may be read contributes 0 and one that may not
        // contributes -1e9. Filling it the other way round -- ones for what is
        // allowed -- type-checks, runs, and answers a question nobody asked.
        engine::Tensor mask({batch, kHeads, take, capacity}, -1e9f, false);
        {
            float* out = mask.data();
            for (std::size_t r = 0; r < batch; ++r) {
                for (std::size_t s = 0; s < take; ++s) {
                    // Row r's new position s sits at at[r]+s and may read every
                    // position up to and including itself.
                    const std::size_t reach = std::min(at[r] + s, capacity - 1);
                    for (std::size_t h_i = 0; h_i < kHeads; ++h_i) {
                        float* row = out + ((r * kHeads + h_i) * take + s) * capacity;
                        std::fill_n(row, reach + 1, 0.0f);
                    }
                }
            }
        }

        for (std::size_t b = 0; b < kBlocks; ++b) h = blocks[b].forward(h, caches[b], mask);
        return head.forward(h);
    }

    // Caches sized for this model, one per block.
    std::vector<engine::nn::KVCache> make_caches(std::size_t rows,
                                                 std::size_t capacity = kSeqLen) const {
        std::vector<engine::nn::KVCache> out;
        out.reserve(kBlocks);
        for (std::size_t i = 0; i < kBlocks; ++i) {
            out.emplace_back(rows, kHeads, capacity, kDModel / kHeads);
        }
        return out;
    }

    std::vector<engine::Tensor> parameters() {
        std::vector<engine::Tensor> out = embedding.parameters();
        for (engine::nn::TransformerBlock& block : blocks) {
            for (const engine::Tensor& p : block.parameters()) out.push_back(p);
        }
        for (const engine::Tensor& p : head.parameters()) out.push_back(p);
        return out;
    }

    // Names have to be stable across builds, because that is what a checkpoint
    // matches on. The block index is in the prefix so adding a third block does
    // not silently renumber the first two.
    std::vector<std::pair<std::string, engine::Tensor>> named_parameters() {
        std::vector<std::pair<std::string, engine::Tensor>> out;
        for (auto& np : embedding.named_parameters("embedding")) out.push_back(np);
        for (std::size_t i = 0; i < blocks.size(); ++i) {
            for (auto& np : blocks[i].named_parameters("block" + std::to_string(i))) {
                out.push_back(np);
            }
        }
        for (auto& np : head.named_parameters("head")) out.push_back(np);
        return out;
    }

    void train(bool mode) {
        embedding.train(mode);
        for (engine::nn::TransformerBlock& block : blocks) block.train(mode);
        head.train(mode);
    }
};

}  // namespace braid

#endif  // BRAID_CHARMODEL_HPP
