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

#include <cstddef>
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
constexpr std::size_t kSeqLen = 256;
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
    engine::Tensor forward(const engine::Tensor& ids, const engine::Tensor& mask) {
        engine::Tensor h = embedding.forward(ids) + positions;
        for (engine::nn::TransformerBlock& block : blocks) h = block.forward(h, &mask);
        return head.forward(h);
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
