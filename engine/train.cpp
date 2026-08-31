// Trains the character model and writes a checkpoint for the server to load.
//
// It began as charlm_demo's training loop with the demo taken off it, and has
// since diverged in the two places a sixty-times-larger model demanded: the
// corpus is both repositories rather than five documents, and the learning rate
// is a schedule rather than a constant.
//
// The server does not train, and nothing in the serving path can be allowed to
// depend on having trained -- so the split is a separate binary rather than a
// flag.
//
//   braid_train <repo-root> <out-prefix> [steps]
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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

// Four, down from sixteen, and the reason is the square in the attention.
//
// Six blocks keep (batch, heads, S, S) scores alive for the backward pass. At a
// 256-id context and a batch of sixteen that was 25 MB a block; at 1024 it is
// 100 MB a block even at four, so 600 MB of scores before the parameters, the
// gradients and Adam's two moments are counted, on a card with 8 GB.
//
// Four is also the batch that keeps the step the same size in tokens: 4 * 1024
// is 16 * 256, so the same number of steps still sees the same amount of corpus
// and the learning rate schedule does not have to be re-derived. What changes is
// the cost of a step, because attention is the part that squares.
constexpr std::size_t kBatch = 4;

// write_vocab puts the alphabet beside the weights. Called before training
// rather than after, so that an interrupted run leaves a checkpoint that can
// still be loaded.
bool write_vocab(const std::string& prefix, const engine::data::CharVocab& vocab) {
    std::string alphabet;
    for (std::size_t i = 0; i < vocab.size(); ++i) alphabet += vocab.symbol(i);

    std::ofstream out(prefix + ".vocab", std::ios::binary);
    if (!out) {
        std::cerr << "could not write " << prefix << ".vocab" << std::endl;
        return false;
    }
    out.write(alphabet.data(), static_cast<std::streamsize>(alphabet.size()));
    return out.good();
}

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
        std::cerr << "usage: braid_train <repo-root> <out-prefix> [steps]\n";
        return 2;
    }
    const std::string repo = argv[1];
    const std::string prefix = argv[2];
    const int steps = argc > 3 ? std::atoi(argv[3]) : 1500;

    engine::manual_seed(42);

    // Everything both repositories have to say about themselves: the engine's
    // sources and documentation, and this server's.
    //
    // Still nothing downloaded and still no licence question, which is why the
    // engine's own demo trains this way. It is about a megabyte, which is far
    // too little for a model this size -- the model will memorise it rather
    // than learn from it, and the generated text is worth reading as evidence
    // that the machinery runs and nothing more. The serving measurements this
    // checkpoint exists for do not care what it learned, only how much
    // arithmetic it takes to run.
    //
    // One tree, and it is this repository -- which carries the engine as a
    // submodule, so both are covered without ever stepping outside. An earlier
    // version walked ".." as well, meaning to reach the server from the engine's
    // directory; run from the repository root that is the user's entire
    // Documents folder, and it crashed trying to read it. A corpus builder that
    // can leave its own repository is a bug whether or not it crashes.
    std::vector<std::string> sources;
    {
        std::error_code ec;
        auto walk = std::filesystem::recursive_directory_iterator(
            repo, std::filesystem::directory_options::skip_permission_denied, ec);
        if (ec) {
            std::cerr << "cannot read " << repo << ": " << ec.message() << "\n";
            return 1;
        }
        for (const auto& entry : walk) {
            if (!entry.is_regular_file()) continue;
            const std::string path = entry.path().generic_string();
            if (path.find("/build") != std::string::npos ||
                path.find("/.git") != std::string::npos ||
                path.find("/models/") != std::string::npos) {
                continue;
            }
            const std::string ext = entry.path().extension().string();
            if (ext == ".cpp" || ext == ".hpp" || ext == ".cu" || ext == ".md" ||
                ext == ".go" || ext == ".py") {
                sources.push_back(path);
            }
        }
    }
    std::sort(sources.begin(), sources.end());  // reproducible order, so the corpus is
    const std::string corpus_text = engine::data::load_text(sources);
    if (corpus_text.size() < braid::kSeqLen * 4) {
        std::cerr << "the corpus is empty or unreadable; nothing to train on\n";
        return 1;
    }

    const engine::data::CharVocab vocab(corpus_text);
    const std::vector<std::size_t> corpus = vocab.encode(corpus_text);
    std::printf("corpus: %zu characters from %zu files, %zu distinct symbols\n",
                corpus.size(), sources.size(), vocab.size());

    braid::CharModel model(vocab.size());
    std::size_t parameters = 0;
    for (const engine::Tensor& p : model.parameters()) parameters += p.size();
    std::printf("model: %zu parameters, context %zu\n", parameters, braid::kSeqLen);

    // 3e-4 with warmup, not the 3e-3 the 172 728-parameter model trained at.
    // That rate on six blocks and ten million parameters diverges immediately --
    // measured, not feared: loss 8.86 at step one and 57.8 by step three, which
    // is the gradient blowing up rather than the model being wrong about
    // anything. The engine already has the usual recipe for Transformers, so
    // this uses it rather than hand-tuning a constant.
    engine::optim::Adam opt(model.parameters(), 3e-4f);
    engine::optim::WarmupCosineLR schedule(opt, static_cast<std::size_t>(steps) / 20,
                                           static_cast<std::size_t>(steps), 1e-5f);
    const engine::Tensor mask = engine::nn::causal_mask(braid::kSeqLen);
    engine::Tensor ids({kBatch, braid::kSeqLen}, 0.0f, false);
    std::vector<std::size_t> targets;
    std::mt19937 rng(1234);

    // Written before the first step, not after the last. A checkpoint is only
    // loadable beside the alphabet it was trained with, and the intermediate
    // checkpoints below would otherwise sit next to a stale vocab from whatever
    // ran here before -- a shape mismatch the engine catches loudly, which is
    // better than silence and still an hour of compute spent on a file nobody
    // can open.
    if (!write_vocab(prefix, vocab)) return 1;

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
        schedule.step();

        if (step == 1 || step % 100 == 0 || step == steps) {
            const double elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            const float nats = loss.data()[0];
            std::printf("  step %5d | loss %.4f | %.2f bits/char | lr %.2e | %.1f s\n", step,
                        nats, nats / std::log(2.0f), opt.learning_rate(), elapsed);
            std::fflush(stdout);
        }

        // A checkpoint every so often, because a run this long is a run that can
        // be interrupted. The first attempt at the 1024-id context died at step
        // 4400 of 6000 -- twenty-four minutes of compute -- when something else
        // on the machine wanted the card, and there was nothing on disk to
        // resume from because the only write was at the end.
        //
        // Written to a temporary name and then renamed, so a crash during the
        // write leaves the previous checkpoint intact rather than a truncated
        // file that loads as far as the corruption.
        if (step % 500 == 0 && step != steps) {
            auto named = model.named_parameters();
            const std::string partial = prefix + ".partial";
            engine::save_parameters(named, partial + ".bin");
            std::error_code ec;
            std::filesystem::rename(partial + ".bin", prefix + ".bin", ec);
            if (ec) {
                std::cerr << "could not move the checkpoint into place: " << ec.message()
                          << std::endl;
            }
        }
    }

    // The alphabet travels with the weights. CharVocab derives its alphabet as
    // the distinct bytes of whatever string it is given, ascending -- so handing
    // it back its own alphabet reconstructs it exactly, and the server needs no
    // access to the corpus the model was trained on.
    auto named = model.named_parameters();
    engine::save_parameters(named, prefix + ".bin");


    std::printf("wrote %s.bin (%zu tensors) and %s.vocab (%zu symbols)\n", prefix.c_str(),
                named.size(), prefix.c_str(), vocab.size());
    return 0;
}
