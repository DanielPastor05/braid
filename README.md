# braid

[![CI](https://github.com/DanielPastor05/braid/actions/workflows/ci.yml/badge.svg)](https://github.com/DanielPastor05/braid/actions/workflows/ci.yml)
![Go 1.27](https://img.shields.io/badge/Go-1.27-00ADD8)
![CUDA 13.3](https://img.shields.io/badge/CUDA-13.3-76B900)
![license](https://img.shields.io/badge/license-MIT-blue)

A continuously batched inference server for
[cpp-ai-engine](https://github.com/DanielPastor05/cpp-ai-engine), written in Go.

Independent requests arrive whenever they arrive and are merged into a single
forward pass. A request that shows up mid-flight joins the batch at the next
step; a request that finishes leaves at the next step; and **nothing about the
batch reaches another sequence through the scheduler** — every row carries its
own history, its own temperature and its own seeded generator.

Whether the *arithmetic* keeps that promise is a separate question with a weaker
answer, and this page tries hard not to blur the two.
[It is measured, not guaranteed](#the-property-that-makes-it-correct).

*Built with heavy AI assistance — [what that means](#how-this-was-built).*

| | |
|---|---|
| **Model** | the character Transformer from `cpp-ai-engine`, 172 728 parameters, 64-id context, 120-symbol alphabet, 1.94 bits/char |
| **Card** | RTX 3060 Ti, CUDA 13.3, engine built with `-DENGINE_CUDA=ON` |
| **Forward passes for the same 7 680 tokens** | 7 680 at one client, **273 at thirty-two** |
| **Throughput** | 988 tokens/s at one client, **7 185 at thirty-two** |
| **Where a step goes at batch 32** | 3.39 ms model, 0.35 ms pipe, 0.06 ms sampling |
| **Two engine thresholds moved** | matmul 2²² → 2²⁰ (**3.5×** at one client), LayerNorm 2¹⁵ → 2 048 (**1.4–1.9×** at one to five) |
| **CPU and CUDA paths** | 200 characters identical, sampled independently |
| **Batching invariance** | measured identical alone and at mean batch 8.68 — a property, not a guarantee |
| **A worker killed mid-load** | 512 requests, **0 failed**, tail unmoved — workers hold no state |
| **A worker hung mid-step** | killed on a deadline and failed over; before that it stopped the server for good |
| **Landed upstream** | three PRs on the engine: the residency clause, the correction to it, and the floor becoming a threshold |

---

## What batching buys, and where it stops paying

128 generations of 60 tokens at each concurrency level — 7 680 tokens every row,
so the step column is directly comparable down the page. Measured after a warm-up
sweep, for reasons two sections below.

| clients | forward passes | mean batch | tokens/s | TTFT p50 | TTFT p95 | wall ms | forward ms | copy ms | sample ms | pipe ms | kernels |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 7 680 | 1.00 | 905 | 4 ms | 7 ms | 1.01 | 0.92 | 0.04 | 0.00 | 0.05 | 60 |
| 2 | 4 161 | 1.85 | 1 806 | 5 ms | 7 ms | 0.99 | 0.89 | 0.04 | 0.00 | 0.05 | 60 |
| 4 | 2 095 | 3.67 | 3 146 | 5 ms | 7 ms | 1.14 | 1.03 | 0.06 | 0.01 | 0.05 | 60 |
| 8 | 1 030 | 7.46 | 4 852 | 7 ms | 9 ms | 1.51 | 1.35 | 0.09 | 0.01 | 0.05 | 60 |
| 16 | 516 | 14.88 | 6 133 | 12 ms | 18 ms | 2.37 | 2.00 | 0.15 | 0.03 | 0.19 | 60 |
| 32 | 258 | 29.77 | **7 181** | 19 ms | 27 ms | 4.01 | 3.14 | 0.33 | 0.07 | 0.47 | 60 |
| 64 | 245 | 31.35 | 6 917 | **286 ms** | 314 ms | 4.32 | 3.06 | 0.47 | 0.11 | 0.68 | 60 |

The kernel column is flat now, and both discontinuities that used to be in this
table are gone. Getting them out took two changes to the engine's dispatch
thresholds and one change that turned out not to matter; the next two sections
are what that cost and what it bought.

**One client needs 7 680 forward passes to produce 7 680 tokens.** That is what
serving requests one at a time means, and the first row confirms the accounting:
exactly one step per token, mean batch 1.00.

**Thirty-two need 258.** Same work, 30× fewer trips through the model, and
eight times the throughput of a single client.

**Sixty-four is the wall.** Throughput stops rising — 7 181 to 6 917, inside the
noise — while the median time to first token goes from 19 ms to 286 ms. With `-max-batch 32`, half the clients
are waiting for a seat at any moment, and waiting is most of what the extra
concurrency buys them. On a throughput chart alone these last two rows look like
progress. Only the tail says otherwise.

### The batch of one was not using the GPU at all

The first version of this table had a hole in it. Throughput climbed — 200, 353,
433 — while the model itself got *slower*: 4.81 ms a step at batch 1, 5.41 at
batch 2, 6.55 at batch 3, and then a collapse to 2.98 at batch 4. A larger batch
cannot compute faster, so something other than batch size was moving, and the
step split could not say what.

So the worker started reporting how many CUDA kernels each forward launched.
**At batch 1 the answer was zero.** The whole forward ran on the CPU. The engine
keeps an operation on the host when it falls below `ENGINE_CUDA_MIN_FLOPS`,
because for a training step the transfer costs more than the arithmetic saves,
and this model's feed-forward at batch 1 is 64 × 96 × 192 × 2 ≈ 2.4 MFLOP
against a 2²² ≈ 4.2 MFLOP default. Batches of two and three straddled the line —
nine and twelve kernels, paying for transfers without earning them, which is the
worst place to be and exactly where the model measured slowest.

That threshold was set for training. Serving is not training, so it was swept
rather than argued about. Batch 1, everything else held:

| `ENGINE_CUDA_MIN_FLOPS` | kernels | forward ms | tokens/s |
|---|---|---|---|
| 2²² = 4 194 304 — the engine's default | **0** | 4.46 | 218 |
| 2²⁰ = 1 048 576 | 48 | **1.20** | **760** |
| 262 144 | 48 | 1.25 | 730 |
| 65 536 | 48 | 1.21 | 756 |
| 1 — everything on the card | 53 | 1.33 | 683 |

The gain arrives all at once between 2²² and 2²⁰ and then stops. Going further
does nothing, and going all the way to 1 makes it slightly *worse*: five more
kernels for five operations that really were better off on the host. The
threshold is not wrong, it is calibrated for a different question.

**braid serves at 2²⁰**, and the table at the top of this page is measured
there. It is worth 3.5× at one client, 3.1× at two, 2.9× at three — and it
removes the dip entirely, leaving throughput rising monotonically from 698 to
7 895. The engine's own default is still one flag away, and `-cuda-min-flops 0`
defers to whatever it was built with.

None of that would have been safe to turn on without knowing it did not change
what the model writes, so `TestTheCPUAndGPUPathsAgree` generates the same 200
characters twice — once at 2²², entirely on the host, once at 2²⁰, on the card —
and compares them. They are identical. Two hundred sequential draws from logits
computed by two different implementations, and not one of them lands
differently.

---

## The second limit: LayerNorm, and why the chain breaks

Lowering the threshold did not flatten the curve. A second discontinuity
remained, and driving the worker at *exact* batch sizes — rather than at client
counts, whose steps are a mixture of sizes — put it between five and six:

| n | kernels | host→device | device→host | forward ms |
|---|---|---|---|---|
| 1 | 48 | 5 | 5 | 1.58 |
| 2 | 48 | 5 | 5 | 1.52 |
| 3 | 48 | 5 | 5 | 1.95 |
| 4 | 48 | 5 | 5 | 2.03 |
| 5 | 48 | 5 | 5 | 2.53 |
| **6** | **60** | **1** | **1** | **1.14** |
| 8 | 60 | 1 | 1 | 1.11 |
| 12 | 60 | 1 | 1 | 1.47 |

Four things it is not, each ruled out by measurement rather than by reading:
`ENGINE_CUDA_MIN_FLOPS`, swept from 2²² down to 98 304 — the jump does not move.
`ENGINE_CUDA_MIN_ELEMENTS`, swept from 2²⁰ down to 1 024 — it changes the kernel
count either side but not where the jump is. And the matmul kernel itself,
pinned in turn to tiled, register-tiled and vectorised — the jump survives all
three.

**The transfer columns are the answer, and the kernel count is only its
shadow.** Below six, a step crosses PCIe five times in each direction. At six
and above, once. The extra kernels are not extra work; they are the same work
staying on the card instead of coming home between operations.

The engine dispatches an operation when it is big enough to be worth the
transfer *or* when its input is already on the device — the second clause being
the one that lets a forward pass chain across the card without touching the
host. `elementwise_worth_it` has it. The matmul entry has it. Every operation
in this model's forward has it except one:

```cpp
// src/cuda/kernels.cu
constexpr size_t kMinLayerNormElements = 1u << 15;

bool layernorm(...) {
    if (!enabled() || x.size() < kMinLayerNormElements) return false;
```

A hard floor, with no residency clause. LayerNorm's input here is
`n × 64 × 96` = `n × 6 144` elements, so it clears 32 768 at
`n = 32 768 / 6 144 = 5.33`, and the first whole batch that clears it is six.
Below that, all four LayerNorms in the model — two blocks, two each — refuse.
Each refusal pulls the activations off the card and the next operation puts them
back: four breaks plus the initial upload is the five crossings in the table.
The constant is a `constexpr`, which is why neither environment variable could
move it.

Then tested rather than left as a good story. Rebuilding the engine with the
floor at 2¹³ = 8 192 predicts a crossover at `8 192 / 6 144 = 1.33`, so at a
batch of two:

| n | kernels | host→device | forward ms |
|---|---|---|---|
| 1 | 48 | 5 | 1.26 |
| **2** | **60** | **1** | **1.04** |
| 5 | 60 | 1 | 0.98 |

It moved exactly there. **The floor is the lever** — 2.6× at five sequences,
2.2× at four, 2.0× at three, on the forward time in this table. What that
becomes once the rest of a step is counted is
[two sections down](#making-the-floor-a-knob-and-what-it-was-worth), and it is
smaller.

### The fix that was not the fix

The obvious repair looked different, and it was wrong. Every other dispatch in
the engine reads "big enough to be worth the transfer **or** already resident on
the device" — `elementwise_worth_it()`, the matmul entry — and LayerNorm was the
only one carrying a bare floor with no residency clause. So the clause was
added, upstream, with a test:
[cpp-ai-engine#2](https://github.com/DanielPastor05/cpp-ai-engine/pull/2).

Measured after merging and updating the submodule pin, it changed **nothing**.
The table was identical: 48 kernels and five crossings each way below six.

The clause only fires when the input is *already* on the card, and in this model
it never is. The chain is broken before it reaches a LayerNorm — the embedding
output starts on the host, and `reshape` and `transpose` are still there too.
What had moved the numbers in the experiment above was lowering the **floor**,
which makes LayerNorm dispatch regardless of residency, paying an upload to hold
the chain afterwards. Two different changes, and the speedup belonged to the
second.
[cpp-ai-engine#3](https://github.com/DanielPastor05/cpp-ai-engine/pull/3)
corrects the claim. The clause stays — it makes LayerNorm consistent with every
other operation, and it will pay the moment the ops around it stop coming home —
but it is latent, and an earlier version of this file credited it with a speedup
it does not deliver.

### Making the floor a knob, and what it was worth

The floor was the lever and it was a `constexpr`, while `ENGINE_CUDA_MIN_FLOPS`
and `ENGINE_CUDA_MIN_ELEMENTS` were both settable and documented as sweepable
"without recompiling". So it became the third:
[cpp-ai-engine#4](https://github.com/DanielPastor05/cpp-ai-engine/pull/4) adds
`ENGINE_CUDA_MIN_LAYERNORM`, `min_layernorm_elements()`, and a third argument to
`set_thresholds`. The engine's default does not change.

**braid serves at 2 048.** Any value under 6 144 puts a single sequence on the
card; that leaves room. End to end, against the engine's default:

| clients | floor 2¹⁵ | floor 2 048 | |
|---|---|---|---|
| 1 | 670 / 693 | 991 / 976 | **1.4×** |
| 2 | 1 006 / 1 060 | 1 709 / 1 789 | **1.7×** |
| 3 | 1 295 / 1 240 | 2 150 / 2 375 | **1.8×** |
| 4 | 1 441 / 1 566 | 2 646 / 2 954 | **1.9×** |
| 5 | 1 674 / 1 703 | 3 220 / 3 238 | **1.9×** |
| 6 | 3 394 / 3 765 | 2 862 / 4 116 | 1.0× |
| 8 | 3 950 / 4 528 | 3 639 / 4 803 | 1.0× |

Two figures per cell because the sweep was run in both directions, for the
[reason given below](#what-the-measurements-needed-before-they-could-be-published).

**Rows six and eight are the control.** Above the default floor both
configurations dispatch identically — 60 kernels, one crossing each way — so
they must show no difference, and they do not. What they show instead is the
size of the noise, ±16%, which is the band every other row has to beat. Rows one
to five beat it in both directions.

1.4× to 1.9×, then, end to end — not the 2.0× to 2.6× that the forward-time
column alone suggested. A step is more than its forward.
---

## Killing a worker under load

`-workers N` puts a pool behind the scheduler. The scheduler does not know: a
`Pool` is a `Backend` like any other, and nothing in `internal/sched` changed to
support this.

That is not luck. **A worker holds no state between steps.** The sequence's
history lives in the scheduler and every step sends the whole 64-id window, so
there is no cache to rebuild and no session to migrate — a step that fails on
one worker is asked of the next, with the same bytes, and the caller cannot tell.
A server that kept the context on the card would have to choose between
rebuilding it elsewhere and failing the request.

512 generations of 200 tokens at sixteen clients, twice, against the same server:

| | requests | tokens/s | TTFT p50 | TTFT p95 | failed |
|---|---|---|---|---|---|
| nothing dies | 512 | 7 246 | 11 ms | 17 ms | 0 |
| one worker killed mid-run | 512 | 7 281 | 11 ms | 18 ms | **0** |

`Stop-Process -Force`, not a signal: a signal would exercise the shutdown path,
which is not the path in question. The server discovers it as a failed write to
a pipe it had every reason to expect to work — which is what an OOM kill or a
driver reset looks like from inside.

**One death, one failover, one restart, and the tail did not move.** The blast
radius of a worker dying is exactly one step, because the scheduler advances the
batch one step at a time and that worker was holding exactly one. It is worth
being clear that this is a consequence of the design rather than a recovery
mechanism that had to be engineered: there was nothing to lose, so nothing was
lost.

### What a pool is not

**It is not capacity.** The scheduler steps serially, so at any instant one
worker is computing and the rest are idle — three workers reach the same
throughput as one, which the table above shows by matching the single-worker
number to within noise. The pool buys redundancy and nothing else on one card.

Making it capacity means splitting a batch across workers and waiting for all of
them, which is worth doing when they are on different GPUs and pointless when
they share one. That is the shape of the next change, not this one.

The chaos test refuses to pass quietly: if the kill lands between steps and no
failover is recorded, it fails rather than reporting a clean run. An earlier
version of the shell harness killed a stray worker left over from an earlier
run, and produced a beautiful table of a death that never happened.

**And it runs in CI**, which took a worker with no model in it. The one above
needs a GPU, a checkpoint and MSVC, so it is skipped on every machine but one —
which left the part of this repository that is about processes rather than
arithmetic verified nowhere that anybody else could see. `internal/backend` now
re-executes its own test binary as a worker that speaks the protocol and holds
nothing, so the pool, the failover, the restart and the frame itself are checked
on every push, on a Linux runner with no card in it. It kills the process the
same way, with the same assertion: every step still returns the answer its
window earns.

---

## Where a step actually goes

The worker times itself between finishing the read and starting the write, and
reports three figures in every response; the server subtracts them from the wall
clock it measured around the whole call. What is left is the pipe — two writes,
two reads, and the serialising at each end. It is a subtraction rather than its
own measurement on purpose: nothing can hide in the gap, because the gap is the
answer.

| | batch 1 | batch 8 | batch 32 |
|---|---|---|---|
| the model's own kernels | 0.92 ms | 1.35 ms | 3.14 ms |
| pulling the logits off the device | 0.04 ms | 0.09 ms | 0.33 ms |
| sampling, 120-way softmax per row | 0.00 ms | 0.01 ms | 0.07 ms |
| **the pipe** | **0.05 ms** | **0.05 ms** | **0.47 ms** |
| pipe as a share of the step | 5.0% | **3.3%** | **12%** |

The pipe costs a twentieth of a millisecond at small batches and about half at
thirty-two, because the frame it carries is `n × 64 × 4` bytes — 256 bytes at
batch 1, 8 KB at batch 32. Its share reaches 12%, and it gets there because the
model's own time grows more slowly than the frame does.

### The copy is mostly waste, and less waste than it looked

`copy` is its own column because of what it pulls. The worker asks for the
logits with `logits.data()`, which brings back `(n, 64, vocab)` — every position
of every window — and then samples from one row of each: the last. **Sixty-three
sixty-fourths of that transfer is for nothing**, on every step.

Sixty-four times too much data sounds like the headline finding, which is why it
was measured before it was described. It is 0.04 ms at batch 1 and 0.33 ms at
batch 32: **between 4% and 8% of a step**. Real, worth removing, and not the
biggest thing wrong with this server. A claim inspected in the source and a
claim measured are different claims, and reading the code was enough to find it
and not enough to size it.

Removing it needs a device-side slice, which the engine does not have —
`Tensor::slice` goes through `data()`, so slicing after the forward copies
everything down and then throws it away. That is a change in
`cpp-ai-engine`, listed under Next with the 8% next to it rather than an
adjective.

Twelve percent is the price of not being able to link the engine into the server.
It is worth knowing rather than assuming, and it is small enough that the
[reason for the process boundary](#why-the-model-runs-in-another-process) still
holds.

---

## What the measurements needed before they could be published

**A row labelled 64 was measuring 48.** With `-requests 48` and 64 clients,
sixteen of them never got a request and the level quietly measured a smaller
number than the one it printed. The harness now refuses a sweep whose request
count is below its concurrency. Correcting it moved the mean batch at 64 clients
from 23.4 to 31.2.

**Levels are measured in order, and the card heats up.** Running the same sweep
descending instead of ascending is the check on that, and most of the table
survives it: 698 against 648 at one client, 1 613 against 1 619 at four, 5 037
against 4 987 at eight, 7 390 against 7 739 at thirty-two. The exception is the
last row, where 7 895 becomes 5 889 — a 25% spread, on the one level whose
requests spend most of their life queued rather than computing, and where
sixty-four client goroutines contend with the server and the worker for the same
host CPU. The 64-client throughput figure should be read as "about seven
thousand, give or take a thousand", and the rest of the column as measured.

**The counts do not move.** Steps, mean batch and kernels per step came out
the same in both directions — 7 680/1.00/48 at one client, ~262/29.3/60 at
thirty-two. They are counts, not timings, and they are what the argument for
batching actually rests on. That is why the table leads with forward passes
rather than with tokens a second.

---

## The property that makes it correct, and how far it goes

Two claims live here and they are not equally strong. Keeping them apart is the
point of this section.

**The scheduler guarantees isolation.** Each sequence carries its own history,
its own temperature and its own seed, and the sampler is a fresh
`mt19937_64(seed)` per row. No row can read another's state, at any batch size,
on any hardware. That is structural and it is what the tests below hold the code
to.

**The arithmetic does not guarantee bit-identical logits.** A batch of *n* is a
different matrix product from a batch of one, and the engine picks its matmul
kernel from `rows = n × 64` with a cut at 128 — so a batch of one and a batch of
two do not even run the same kernel. Different reduction order, different last
bit. The sampler draws by walking an inverse CDF, so a one-ULP difference near a
boundary flips a token, and from there the sequences part.

So the honest statement is: **isolation is guaranteed, identity is measured.**

`TestBatchingDoesNotChangeOutput` runs against the mock, whose next token is a
hash of the entire window: run a request alone, keep the text, run it again
while eight neighbours join and leave around it on a backend with different
timings, compare character for character. That one is bit-exact by construction
and catches every scheduler bug — a crossed row, a window padded from the wrong
end, the wrong sequence advanced.

`TestBatchingDoesNotChangeOutputOnTheRealModel` runs the same shape against the
engine. **On this model and this card it comes out identical**: 120 characters,
at a mean batch of 8.68 over 121 steps. One model, one card, one seed, 120
draws. That is a strong empirical result and it is not a proof, and an earlier
version of this page opened by calling it one.

What would make it a guarantee is padding every batch to a fixed width so the
kernel and the reduction order never change — at the cost of computing rows
nobody asked for. What would make the measurement honest at scale is a
divergence *rate* over 10⁵ draws across batch sizes rather than one comparison.
The second is the more interesting number and neither is done.

`TestTheCPUAndGPUPathsAgree` is the sharper version of the same idea, and it
exists because lowering the matmul threshold had to be safe before it could be a
default: the same 200 characters generated once entirely on the host and once on
the card, by two different implementations of the same model. Identical. Two
hundred sequential draws from independently computed logits, none landing
differently.

Both batching tests also assert the sequences really did share steps. An earlier
version of the first passed while batching nothing at all — the backend was fast
enough that each request finished before the next arrived, mean batch 1.00,
invariant trivially satisfied, nothing proved.

The GPU tests skip unless `BRAID_WORKER` and `BRAID_MODEL` are set, which CI has
neither of. If they are set and wrong, they fail rather than skip: an earlier
version skipped on a bad path and reported a green run for a test that never
started.

---

## The failure that a dead process cannot stand in for

A worker that dies closes its pipe and the next read fails at once. That is the
easy failure, it is what the chaos tests kill, and it was the only one this
server survived.

A worker that is **alive and silent** — a wedged kernel, a driver reset that
leaves the process up — holds the pipe open, and a pipe read has no timeout of
its own. `Backend.Step` had no context and no deadline, so that read blocked the
scheduler's single goroutine forever: every request behind it stalled, every new
one queued, and `/healthz` went on answering that the server was fine. It was
the worst bug in this repository and nothing in it could have found it, because
every test killed processes rather than freezing them.

`Step` now takes a context and each worker enforces its own ceiling. On expiry
the process is killed rather than abandoned, and that is not tidiness: the
protocol carries no request identifier, so a worker that answered late would
answer the *next* step with this step's ids and nothing could notice. Killing it
turns a hang into a death, which is a failure the pool already knows how to
handle — failed over, restarted, transparent.

Two tests hold it, both in CI, neither needing a GPU. The fake worker grew a
`hang` mode: read the frame, answer never, stay alive. One asserts a single
worker gives up on time and that the process is gone afterwards — checked by
closing it, since there is no portable "is this pid alive", and a `Close` that
returns promptly is proof the kill landed. The other hangs every worker in a
pool at once, which is the correlated failure that actually happens when they
share a card, and asserts the server digs itself out rather than staying dead.

Writing the `hang` mode took two attempts. The first parked the fake in
`select{}`, which the Go runtime calls a deadlock and answers by killing the
process — producing the immediate EOF the mode existed to avoid. The assertion
caught it. A weaker one would have passed.

---


## Why the model runs in another process

`nvcc` on Windows compiles only through MSVC. cgo links only through a
GCC-compatible toolchain. A C++ static library from one is not linkable by the
other, so the engine cannot be linked into the Go binary at all.

A pipe has no ABI to disagree about. `braid_worker` is a C++ process that owns
the model and answers step requests over stdin and stdout in a fixed binary
frame; the Go side writes `n` windows and reads `n` ids back, plus the three
timings and the kernel count it needs to say where the step went. It costs
[about twelve percent at batch 32](#where-a-step-actually-goes).

What falls out of it is the shape the next phase needs: a worker is a thing that
can be killed.

---

## Three decisions worth the words

**The queue rejects rather than grows.** Past `-queue` the answer is HTTP 429
with a `Retry-After`. An unbounded queue does not make a server faster; it
converts a throughput problem into a latency problem and reports neither.

**Deadlines are checked at admission, not at completion.** A caller that sets
`max_wait_ms` and waits longer is rejected *before* the first forward pass.
Rejecting early costs nothing; discovering it after the GPU has paid for the
work wastes the work.

**A stream's buffer is exactly `max_tokens` long, so the loop's send can never
block.** The first version capped it at 32 tokens and killed whoever fell
behind. That punished a caller for being briefly behind — HTTP clients are
bursty — and let one slow reader take out its own request for no good reason.
Sizing the buffer to the most a request could ever produce removes the question:
a caller that reads nothing at all still completes, buffered, paying for itself
in memory and costing its batch neighbours nothing. `MaxTokensLimit` bounds that
memory, and bounds how long one sequence can hold a seat in the batch — the same
number doing both jobs.

---

## What the server says about itself

`/stats` reports the scheduler's counters, the step breakdown when the backend
can produce one, the pool when there is one, and **its own latency**:

```json
{"samples": 1024, "window": 1024,
 "ttft_p50_ms": 2.659, "ttft_p95_ms": 250.669, "ttft_p99_ms": 278.05,
 "total_p50_ms": 125.92, "total_p95_ms": 508.747, "total_p99_ms": 544.796}
```

That last part was missing for longer than it should have been. A comment in
`stats.go` said *"a server that computes its own averages is a server that hides
its tail"* — and the file it was written in exposed counters and a mean batch
size. The percentiles lived only in the load harness, which meant this server
could describe its own tail only while somebody was benchmarking it, which is
exactly when a tail matters least.

A window of the last 1 024 completions rather than a lifetime, because a p99
over every request since boot is a number an hour of good service cannot move
and an incident an hour ago never leaves. Rejections are kept out of it: they
have no time to first token, and counting them as zero would drag every
percentile down precisely when the server is overloaded and the tail is the only
thing worth reading.

---

## How this was built

With heavy AI assistance, the same as
[cpp-ai-engine](https://github.com/DanielPastor05/cpp-ai-engine#how-this-was-built),
and it is said here for the same reason: a reader who works it out for
themselves has been told something the author would rather they had not noticed.

What that does and does not mean, as precisely as it can be put:

The design decisions are mine and they are the part worth reading — a stateless
worker so failover is a retry, one goroutine so the batch has no locking to get
wrong, a `Backend` seam narrow enough to test the scheduler with no model
present. So is the discipline the measurements are held to: every number on this
page was produced by running the thing, several of them contradicted what I
expected, and the ones that turned out wrong were corrected in place rather than
quietly replaced.

Three of those corrections are still on this page on purpose. A speedup credited
to the wrong change and fixed upstream in a second pull request. A 2.6× that was
forward-time only and is 1.9× end to end. A 64× over-copy that reads like a
catastrophe in the source and measures at 8%. Leaving them visible is the only
way the rest of the numbers mean anything.

---

## Layout

```
cmd/braid/          the server
cmd/braidload/      the load harness that printed the tables
internal/sched/     the loop: admission, batching, cancellation, stats
internal/backend/   the seam -- Backend is six methods; Mock, Worker and Pool implement them
internal/api/       HTTP and server-sent events
engine/             the C++ side: the model, the trainer, the worker process
third_party/        cpp-ai-engine, pinned as a submodule
```

`internal/sched` imports nothing that knows what a GPU is. The whole batching
argument is testable, and tested, with no model present.

## Building it

The Go server needs nothing but Go. The worker needs MSVC, CUDA and the
submodule:

```bash
git clone --recurse-submodules https://github.com/DanielPastor05/braid
```

```bash
cmake -B build -S engine -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build
```

```bash
./build/braid_train third_party/cpp-ai-engine models/charlm 1500
```

Training is 31 seconds on a 3060 Ti and lands at 1.94 bits/char against 6.91 for
a uniform guess over the alphabet. The checkpoint is not committed because
`braid_train` is seeded and reproduces it.

```bash
go run ./cmd/braid -addr 127.0.0.1:8420 -worker ./build/braid_worker.exe -model models/charlm
```

```bash
go run ./cmd/braidload -addr http://127.0.0.1:8420 -requests 128 -max-tokens 60
```

Without `-worker`, the server runs a mock backend and says so on every startup,
because a server that quietly served plausible nonsense at good latencies is a
server whose numbers end up pasted somewhere.

---

## Next

Ordered by what a measurement says, not by what sounds impressive.

1. **A device-side slice, in the engine.** The worker pulls `(n, 64, vocab)` off
   the card and reads one row of each. Worth
   [4% to 8% of a step](#the-copy-is-mostly-waste-and-less-waste-than-it-looked),
   and it cannot be done from here: `Tensor::slice` goes through `data()`.
2. **Serve a model that is not a toy.** 172 728 parameters, a 64-id context and
   a 120-character vocabulary mean the GPU is never the constraint and none of
   these numbers transfer to anything larger. Everything else on this list is
   worth less than this one.
3. **A KV cache**, and publishing what it costs: the stateless worker is what
   makes failover a retry, and a cache is state. That trade, measured, is more
   interesting than the speedup.
4. **A divergence rate for the batching invariant**, over 10⁵ draws across batch
   sizes, instead of one 120-character comparison. Turns a strong empirical
   claim into a number.
5. **Split a batch across workers.** Today a pool is redundancy and not
   capacity; splitting is worth doing across GPUs and pointless on one, so it
   wants a second card before it wants code.
6. **Move the load generator off the machine under test.** The 64-client row is
   the only figure here with a spread worth apologising for, and its sixty-four
   goroutines share a CPU with the server and the worker.
7. **Get the embedding and the positional add onto the card.** They are what
   leaves the first LayerNorm's input on the host, which is why the residency
   clause upstream is still latent and the floor has to be lowered instead.

Not on this list, deliberately: authentication, TLS and rate limiting. This
serves a character model on a desk, and adding them would be theatre — the
honest version is that it should not be exposed to anything, and that is a
sentence rather than a feature.
