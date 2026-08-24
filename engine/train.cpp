// Trains the character model and writes a checkpoint for the server to load.
//
// This is charlm_demo's training loop with the demo taken off it: same corpus,
// same optimiser, same hyperparameters, and at the end two files instead of a
// wall of generated text. The server does not train, and nothing in the serving
// path can be allowed to depend on having trained -- so the split is a separate
// binary rather than a flag.
//
//   braid_train <engine-repo-dir> <out-prefix> [steps]
//
// writes <out-prefix>.bin (the weights) and <out-prefix>.vocab (the alphabet).

#include "charmodel.hpp"

#include "engine/autograd.hpp"
#include "engine/data.hpp"
#include "engine/nn.hpp"
#include "engine/optim.hpp"
#include "engine/random.hpp"
#include "engine/serialize.hpp"
#include "engine/tensor.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kBatch = 32;

// One batch of (context, next character) pairs, drawn at random positions.
void sample_batch(const std::vector<std::size_t>& corpus, std::mt19937& rng, engine::Tensor& ids,
                  std::vector<std::size_t>& targets) {
    std::uniform_int_distribution<std::size_t> start(0, corpus.size() - braid::kSeqLen - 2);
    targets.clear();
    targets.reserve(kBatch * braid::kSeqLen);
    float* id_values = ids.data();

    for (std::size_t b = 0; b < kBatch; ++b) {
        const std::size_t begin = start(rng);
        for (std::size_t t = 0; t < braid::kSeqLen; ++t) {
            id_values[b * braid::kSeqLen + t] = static_cast<float>(corpus[begin + t]);
            targets.push_back(corpus[begin + t + 1]);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: braid_train <engine-repo-dir> <out-prefix> [steps]\n";
        return 2;
    }
    const std::string root = argv[1];
    const std::string prefix = argv[2];
    const int steps = argc > 3 ? std::atoi(argv[3]) : 1500;

    engine::manual_seed(42);

    const std::string corpus_text = engine::data::load_text(
        {root + "/docs/DESIGN.md", root + "/docs/PERFORMANCE.md", root + "/docs/ENGINEERING.md",
         root + "/docs/CUDA.md", root + "/README.md"});
    if (corpus_text.size() < braid::kSeqLen * 4) {
        std::cerr << "the corpus is empty or unreadable; nothing to train on\n";
        return 1;
    }

    const engine::data::CharVocab vocab(corpus_text);
    const std::vector<std::size_t> corpus = vocab.encode(corpus_text);
    std::printf("corpus: %zu characters, %zu distinct\n", corpus.size(), vocab.size());

    braid::CharModel model(vocab.size());
    std::size_t parameters = 0;
    for (const engine::Tensor& p : model.parameters()) parameters += p.size();
    std::printf("model: %zu parameters, context %zu\n", parameters, braid::kSeqLen);

    engine::optim::Adam opt(model.parameters(), 0.003f);
    const engine::Tensor mask = engine::nn::causal_mask(braid::kSeqLen);
    engine::Tensor ids({kBatch, braid::kSeqLen}, 0.0f, false);
    std::vector<std::size_t> targets;
    std::mt19937 rng(1234);

    const auto started = std::chrono::steady_clock::now();
    for (int step = 1; step <= steps; ++step) {
        sample_batch(corpus, rng, ids, targets);

        opt.zero_grad();
        const engine::Tensor logits = model.forward(ids, mask);
        const engine::Tensor flat = logits.reshape({kBatch * braid::kSeqLen, vocab.size()});
        engine::Tensor loss = engine::nn::cross_entropy_loss(flat, targets);
        loss.backward();
        engine::optim::clip_grad_norm(model.parameters(), 1.0f);
        opt.step();

        if (step == 1 || step % 100 == 0 || step == steps) {
            const double elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            const float nats = loss.data()[0];
            std::printf("  step %5d | loss %.4f | %.2f bits/char | %.1f s\n", step, nats,
                        nats / std::log(2.0f), elapsed);
            std::fflush(stdout);
        }
    }

    // The alphabet travels with the weights. CharVocab derives its alphabet as
    // the distinct bytes of whatever string it is given, ascending -- so handing
    // it back its own alphabet reconstructs it exactly, and the server needs no
    // access to the corpus the model was trained on.
    auto named = model.named_parameters();
    engine::save_parameters(named, prefix + ".bin");

    std::string alphabet;
    for (std::size_t i = 0; i < vocab.size(); ++i) alphabet += vocab.symbol(i);
    std::ofstream vocab_file(prefix + ".vocab", std::ios::binary);
    if (!vocab_file) {
        std::cerr << "could not write " << prefix << ".vocab\n";
        return 1;
    }
    vocab_file.write(alphabet.data(), static_cast<std::streamsize>(alphabet.size()));
    vocab_file.close();

    std::printf("wrote %s.bin (%zu tensors) and %s.vocab (%zu symbols)\n", prefix.c_str(),
                named.size(), prefix.c_str(), alphabet.size());
    return 0;
}
