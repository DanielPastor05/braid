# braid

A continuously batched inference server for
[cpp-ai-engine](https://github.com/DanielPastor05/cpp-ai-engine), written in Go.

Independent requests arrive whenever they arrive and are merged into a single
forward pass. A request that shows up mid-flight joins the batch at the next
step; a request that finishes leaves at the next step; **neither event changes
one character of what anybody else generates** — and that is checked against the
real model on the GPU, not only asserted.

| | |
|---|---|
| **Model** | the character Transformer from `cpp-ai-engine`, 172 728 parameters, 64-id context, 120-symbol alphabet, 1.94 bits/char |
| **Card** | RTX 3060 Ti, CUDA 13.3, engine built with `-DENGINE_CUDA=ON` |
| **Forward passes for the same 7 680 tokens** | 7 680 at one client, **264 at thirty-two** |
| **Throughput** | 698 tokens/s at one client, **7 390 at thirty-two** |
| **Where a step goes at batch 32** | 3.31 ms model, 0.46 ms pipe, 0.07 ms sampling |
| **The engine's CUDA threshold** | lowered from 2²² to 2²⁰ after measuring: **3.5× at one client** |
| **CPU and CUDA paths** | 200 characters identical, sampled independently |
| **Batching invariance** | identical output alone and at mean batch 8.68, on the GPU |

---

## What batching buys, and where it stops paying

128 generations of 60 tokens at each concurrency level — 7 680 tokens every row,
so the step column is directly comparable down the page. Measured after a warm-up
sweep, for reasons two sections below.

| clients | forward passes | mean batch | tokens/s | TTFT p50 | TTFT p95 | wall ms | forward ms | sample ms | pipe ms | kernels |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 7 680 | 1.00 | 698 | 4 ms | 6 ms | 1.35 | 1.31 | 0.00 | 0.04 | 48 |
| 2 | 4 092 | 1.88 | 1 089 | 6 ms | 8 ms | 1.70 | 1.66 | 0.00 | 0.04 | 48 |
| 3 | 2 723 | 2.82 | 1 306 | 6 ms | 8 ms | 2.13 | 2.09 | 0.01 | 0.04 | 48 |
| 4 | 2 027 | 3.79 | 1 613 | 6 ms | 8 ms | 2.33 | 2.29 | 0.01 | 0.04 | 48 |
| 6 | 1 385 | 5.55 | 3 934 | 6 ms | 8 ms | 1.39 | 1.35 | 0.01 | 0.04 | 58 |
| 8 | 1 016 | 7.56 | 5 037 | 7 ms | 8 ms | 1.47 | 1.43 | 0.01 | 0.03 | 59 |
| 16 | 506 | 15.18 | 6 614 | 11 ms | 14 ms | 2.21 | 2.10 | 0.02 | 0.08 | 60 |
| 32 | 264 | 29.09 | **7 390** | 22 ms | 36 ms | 3.84 | 3.31 | 0.07 | 0.46 | 60 |
| 64 | 244 | 31.48 | 7 895 | **237 ms** | 276 ms | 3.80 | 3.27 | 0.08 | 0.45 | 60 |

**One client needs 7 680 forward passes to produce 7 680 tokens.** That is what
serving requests one at a time means, and the first row confirms the accounting:
exactly one step per token, mean batch 1.00.

**Thirty-two need 264.** Same work, 29× fewer trips through the model, and
ten times the throughput of a single client.

**Sixty-four is the wall.** Throughput is still creeping up, but the median time
to first token goes from 22 ms to 237 ms. With `-max-batch 32`, half the clients
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

**The same thing is still happening one level up, smaller.** Batches of one to
four launch 48 kernels; batch 6 launches 58, and its forward drops from 2.29 ms
to 1.35. Some second class of operation crosses its own line between four and
six. Lowering the threshold further does not move it, so it is a different limit
and it has not been chased.

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
| model, including the copy off the device | 1.31 ms | 1.43 ms | 3.31 ms |
| sampling, 120-way softmax per row | 0.00 ms | 0.01 ms | 0.07 ms |
| filling the (n, 64) tensor | 0.00 ms | 0.00 ms | 0.01 ms |
| **the pipe** | **0.04 ms** | **0.03 ms** | **0.46 ms** |
| pipe as a share of the step | 3.0% | **2.0%** | **12%** |

The pipe costs a twenty-fifth of a millisecond at small batches and half a
millisecond at thirty-two, because the frame it carries is `n × 64 × 4` bytes —
256 bytes at batch 1, 8 KB at batch 32. As a share it reaches 12%, and it gets
there because the model's own time grows more slowly than the frame does.

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

## The property that makes it correct

Continuous batching is only worth anything if a sequence cannot tell it
happened. The window is 64 ids wide, so a batch of *n* is one `(n, 64)` tensor,
and a scheduler bug — a window padded from the wrong end, two rows crossed, the
wrong sequence advanced — produces text that is subtly wrong rather than an
error that is obviously wrong. Nothing crashes. The output is just not what that
request asked for.

So it is tested directly, twice.

`TestBatchingDoesNotChangeOutput` runs against the mock, whose next token is a
hash of the entire window: run a request alone, keep the text, run it again
while eight neighbours join and leave around it on a backend with different
timings, and compare character for character.

`TestBatchingDoesNotChangeOutputOnTheRealModel` runs the same shape against the
engine, and it exists because the mock cannot fail the way the GPU can. A matrix
product tiled for a batch of one need not reduce in the same order as the same
product tiled for a batch of thirty-two, and two logits within a float of each
other could then land on different sides of the sampling threshold — a
divergence from arithmetic rather than from a bug. **On this model and this card
it does not happen**: 120 characters, identical, at a mean batch of 8.68 over
121 steps.

Given the section above, that comparison was sharper than it was designed to be
when it was written: at the time, the run on its own executed on the CPU and the
batched run executed on the card, so it was quietly comparing two arithmetic
implementations. Lowering the threshold put both runs on the card and turned it
back into a test of the scheduler alone — which is what it was meant to be, and
why the CPU-versus-CUDA comparison now has a test of its own rather than
happening by accident.

Both tests also assert that the sequences really did share steps. An earlier
version of the first one passed while batching nothing at all — the backend was
fast enough that each request finished before the next arrived, mean batch 1.00,
invariant trivially satisfied, nothing proved.

The GPU tests skip unless `BRAID_WORKER` and `BRAID_MODEL` are set, which CI has
neither of. If they are set and wrong, the tests fail rather than skip: an
earlier version skipped on a bad path and reported a green run for a test that
never started.

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

## Layout

```
cmd/braid/          the server
cmd/braidload/      the load harness that printed the tables
internal/sched/     the loop: admission, batching, cancellation, stats
internal/backend/   the seam -- Backend is six methods; Mock and Worker implement them
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

1. **A KV cache.** The engine recomputes the full 64-id window every step. Now
   that a step is broken down there is something to measure a cache against, and
   the kernel counts say where it would and would not help.
2. **A router and more than one worker**, then `kill -9` one under load and
   publish the recovery curve.
3. **Find the second threshold.** Something still moves between batch four and
   batch six — 48 kernels to 58, and a forward that drops by 40%. It is not
   `ENGINE_CUDA_MIN_FLOPS`, because lowering that further does not touch it.
4. **Make the 64-client row reproducible.** It is the only figure on this page
   with a spread worth apologising for, and running the load generator off the
   machine under test is the obvious first thing to try.
