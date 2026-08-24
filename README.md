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
| **Throughput** | 196 tokens/s at one client, **5 498 at thirty-two** |
| **Forward passes for the same 7 680 tokens** | 7 680 at one client, **257 at thirty-two** |
| **Where it stops paying** | 64 clients: throughput −0.4%, time to first token ×13.6 |
| **Batching invariance** | identical output alone and at mean batch 8.68, on the GPU |

---

## What batching buys, and where it stops paying

128 generations of 60 tokens at each concurrency level — 7 680 tokens every row,
so the step column is directly comparable down the page. Measured after a
warm-up sweep, for a reason the next section gets to.

| clients | forward passes | mean batch | tokens/s | TTFT p50 | TTFT p95 | total p50 |
|---|---|---|---|---|---|---|
| 1 | 7 680 | 1.00 | 196 | 8 ms | 10 ms | 302 ms |
| 2 | 3 965 | 1.94 | 265 | 15 ms | 18 ms | 455 ms |
| 4 | 1 958 | 3.92 | 775 | 13 ms | 22 ms | 307 ms |
| 8 | 980 | 7.84 | 2 396 | 14 ms | 19 ms | 206 ms |
| 16 | 506 | 15.18 | 4 067 | 16 ms | 23 ms | 233 ms |
| 32 | 257 | 29.88 | **5 498** | 26 ms | 40 ms | 343 ms |
| 64 | 246 | 31.22 | 5 476 | **353 ms** | 398 ms | 678 ms |

**One client needs 7 680 forward passes to produce 7 680 tokens.** That is what
serving requests one at a time means, and the first row confirms the accounting:
exactly one step per token, mean batch 1.00.

**Thirty-two need 257.** Same work, 29.9× fewer trips through the model, 28×
the throughput.

**Two clients is a loss, and it reproduced every time.** Throughput rises to
265 tokens a second, but `total p50` goes from 302 ms to 455 ms — each request
takes half again as long as it would have alone. A batch-2 step costs 7.3 ms
where a batch-1 step costs 5.1 ms: the compute doubles while the fixed cost per
step stays put, so at two the amortisation has not yet caught up with what it
has to pay for. Batching starts winning at four. A README that only plotted
throughput would have shown this row going up and called it a success.

**Sixty-four is the wall.** Throughput moves −0.4% — 5 498 to 5 476 — and the
median time to first token goes from 26 ms to 353 ms. With `-max-batch 32`,
half the clients are waiting for a seat at any moment, and waiting is all the
extra concurrency buys them. Rows 6 and 7 look equally good on a throughput
chart. They are not, and only the tail says so.

Reproduce it:

```bash
go run ./cmd/braid -addr 127.0.0.1:8420 -worker ./build/braid_worker.exe -model models/charlm
```

```bash
go run ./cmd/braidload -addr http://127.0.0.1:8420 -requests 128 -max-tokens 60
```

### Two things the harness had to be fixed to say honestly

**A row labelled 64 was measuring 48.** With `-requests 48` and 64 clients,
sixteen of them never got a request and the level quietly measured a smaller
number than the one it printed. The harness now refuses a sweep where the
request count is below the concurrency, rather than producing a plausible row.
Correcting it moved the mean batch at 64 clients from 23.4 to 31.2.

**The card slows down as it heats up.** The same sweep three times in a row
gave 5 346, then 4 299, then 3 949 tokens a second at 32 clients — a 26% fall
with nothing changed but the temperature. The step counts were identical to the
unit across all three, because those are counts and not timings. So the table
above is taken after a warm-up sweep, which makes it the conservative number
rather than the flattering one, and throughput here is only ever compared within
a single run.

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
it does not happen**: 120 characters, identical, at a mean batch of 8.68 over 121 steps. That
is a measurement about one model on one card, not a guarantee about GPUs.

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
frame; the Go side writes `n` windows and reads `n` ids back. The cost is one
round trip per step, and it is inside the 5.1 ms a batch-1 step takes here — not
yet separated out from the GPU time, which is the next thing to measure rather
than to estimate.

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
cmd/braidload/      the load harness that printed the table
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

Without `-worker`, the server runs a mock backend and says so on every startup,
because a server that quietly served plausible nonsense at good latencies is a
server whose numbers end up pasted somewhere.

---

## Next

1. **Separate the pipe cost from the GPU cost.** Right now both are inside the
   5.1 ms, and "probably about a percent" is not a measurement.
2. **A KV cache.** The engine recomputes the full 64-id window every step, which
   is the honest baseline to measure a cache against.
3. **A router and more than one worker**, then `kill -9` one under load and
   publish the recovery curve.
