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
| **Forward passes for the same 7 680 tokens** | 7 680 at one client, **260 at thirty-two** |
| **Throughput** | ~200 tokens/s at one client, **5 500–6 500 at thirty-two** |
| **Where a step goes at batch 32** | 3.86 ms model, 0.46 ms pipe, 0.07 ms sampling |
| **Where a step goes at batch 1** | 4.81 ms model — **on the CPU, zero kernels launched** |
| **Batching invariance** | identical output alone and at mean batch 8.68, on the GPU |

---

## What batching buys, and where it stops paying

128 generations of 60 tokens at each concurrency level — 7 680 tokens every row,
so the step column is directly comparable down the page. Measured after a warm-up
sweep, for reasons two sections below.

| clients | forward passes | mean batch | tokens/s | TTFT p50 | TTFT p95 | wall ms | forward ms | sample ms | pipe ms | kernels |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 7 680 | 1.00 | 200 | 8 ms | 10 ms | 4.88 | 4.81 | 0.00 | 0.06 | **0** |
| 2 | 3 964 | 1.94 | 353 | 13 ms | 15 ms | 5.45 | 5.41 | 0.00 | 0.04 | 9 |
| 3 | 2 660 | 2.89 | 433 | 14 ms | 17 ms | 6.58 | 6.55 | 0.01 | 0.02 | 12 |
| 4 | 1 971 | 3.90 | 1 265 | 11 ms | 17 ms | 3.03 | 2.98 | 0.01 | 0.03 | 46 |
| 6 | 1 342 | 5.72 | 2 363 | 10 ms | 15 ms | 2.31 | 2.22 | 0.01 | 0.08 | 57 |
| 8 | 995 | 7.72 | 3 098 | 11 ms | 17 ms | 2.38 | 2.31 | 0.02 | 0.06 | 58 |
| 16 | 506 | 15.18 | 4 433 | 15 ms | 25 ms | 3.32 | 3.12 | 0.03 | 0.16 | 59 |
| 32 | 260 | 29.54 | **6 456** | 21 ms | 33 ms | 4.40 | 3.86 | 0.07 | 0.46 | 59 |
| 64 | 244 | 31.48 | 4 615 | **367 ms** | 494 ms | 6.64 | 5.84 | 0.11 | 0.67 | 59 |

**One client needs 7 680 forward passes to produce 7 680 tokens.** That is what
serving requests one at a time means, and the first row confirms the accounting:
exactly one step per token, mean batch 1.00.

**Thirty-two need 260.** Same work, 29.5× fewer trips through the model.

**Sixty-four is the wall.** The median time to first token goes from 21 ms to
367 ms. With `-max-batch 32`, half the clients are waiting for a seat at any
moment, and waiting is all the extra concurrency buys them. On a throughput
chart alone these last two rows look comparable. Only the tail says otherwise.

### Two and three clients are slower than one, and the reason is not batching

Throughput climbs — 200, 353, 433 — but the model itself gets *slower*: 4.81 ms
a step at batch 1, 5.41 at batch 2, 6.55 at batch 3, and then it collapses to
2.98 at batch 4. A larger batch cannot compute faster, so something other than
batch size is moving.

The kernel column is what it is. **At batch 1 the forward launches zero CUDA
kernels**: the whole thing runs on the CPU. The engine has a size threshold —
`ENGINE_CUDA_MIN_FLOPS`, 2²² by default — below which an operation stays on the
host because the transfer costs more than the arithmetic saves. This model's
feed-forward at batch 1 is 64 × 96 × 192 × 2 ≈ 2.4 MFLOP, under the line. At
batch 2 and 3 it straddles it: nine and twelve kernels, enough to pay for
transfers and not enough to be worth them, which is the worst place to be and
exactly where the measurement says the model is slowest. At batch 4 it clears
the threshold, jumps to 46 kernels, and the forward time halves. From batch 6 it
saturates at 57–59 and never changes again.

So the crossover at four is not the scheduler amortising a fixed cost. It is the
engine's own CUDA threshold, rediscovered from the outside by a server that had
no idea it was there. An earlier version of this file explained the dip as
amortisation, which was a plausible story fitted to a number, and wrong.

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
| model, including the copy off the device | 4.81 ms | 2.31 ms | 3.86 ms |
| sampling, 120-way softmax per row | 0.00 ms | 0.02 ms | 0.07 ms |
| filling the (n, 64) tensor | 0.00 ms | 0.00 ms | 0.01 ms |
| **the pipe** | **0.06 ms** | **0.06 ms** | **0.46 ms** |
| pipe as a share of the step | **1.2%** | 2.5% | **10.5%** |

The pipe costs about a twentieth of a millisecond at small batches and about half
a millisecond at thirty-two, because the frame it carries is `n × 64 × 4` bytes —
256 bytes at batch 1, 8 KB at batch 32. As a share it grows from 1% to 10%, and
it grows because the model's time does *not* grow with it: the forward is nearly
flat from batch 6 up, so most of what is added past that point is transport.

Ten percent is the price of not being able to link the engine into the server.
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
descending instead of ascending moves the 64-client throughput from 4 615 to
5 792 — 25%, from nothing but being measured on a cooler card. The 32-client row
moves the other way, 6 456 to 5 516. Every throughput figure on this page
therefore carries something like a quarter of its value in ordering noise, and
the only honest way to read that column is by its shape.

**The counts do not move.** Steps, mean batch and kernels per step came out
identical in both directions — 7 680/1.00/0 and ~260/29.5/59 either way. They
are counts, not timings, and they are what the argument for batching actually
rests on. That is why the table leads with forward passes rather than with
tokens a second.

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

Given the section above, that comparison turns out to be sharper than it was
designed to be: the run on its own executed on the CPU and the batched run
executed on the card, so what it actually shows is agreement across two
different arithmetic implementations. It remains a measurement about one model
on one card, not a guarantee about GPUs.

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
[about ten percent at batch 32](#where-a-step-actually-goes).

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
3. **Get the small batches onto the card anyway.** One to three sequences run on
   the CPU because each *operation* falls below the engine's threshold. Whether
   that threshold — set for training steps — is the right one for a server is a
   question it was never asked.
