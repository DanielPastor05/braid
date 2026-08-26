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
batch reaches another sequence through the scheduler** - every row carries its
own history, its own temperature and its own seeded generator.

Whether the *arithmetic* keeps that promise is a separate question with a weaker
answer, and this page tries hard not to blur the two.
[It is measured, not guaranteed](#the-property-that-makes-it-correct-and-how-far-it-goes).

*Built with heavy AI assistance - [what that means](#how-this-was-built).*

| | |
|---|---|
| **Model** | the Transformer from `cpp-ai-engine`, **10 758 289 parameters**, 6 blocks, 384 wide, 256-id context, 145-symbol byte alphabet, 0.97 bits/char |
| **Card** | RTX 3060 Ti, CUDA 13.3, engine built with `-DENGINE_CUDA=ON` |
| **Throughput** | 144 tokens/s at one client, **354 at sixteen** - median of three |
| **What batching buys** | **2.5x**, and it saturates at sixteen clients |
| **Where a step goes at batch 32** | 81.8 ms model, 1.9 ms copy back, 0.06 ms sampling |
| **Batching invariance** | 0 divergences in 25 000 draws - a bound, not a guarantee |
| **A worker killed mid-load** | 0 requests failed; workers hold no state |
| **A worker hung mid-step** | killed on a deadline and failed over; before that it stopped the server for good |
| **Landed upstream** | three PRs on the engine, all measured, one of them a correction to another |
| **The bug none of that caught** | every window was padded at the wrong end, and [no measurement here could have told me](#the-bug-no-number-could-have-shown-me) |

---

## What batching buys, and where it stops paying

96 generations of 30 tokens at each concurrency level. Every row is the median
of three sweeps, with the throughput range beside it.

| clients | forward passes | mean batch | tokens/s | TTFT p50 | TTFT p95 | wall ms | forward ms | copy ms |
|---|---|---|---|---|---|---|---|---|
| 1 | 2 880 | 1.00 | 144 (143-144) | 10 ms | 11 ms | 6.68 | 6.62 | 0.09 |
| 2 | 1 485 | 1.94 | 221 (221-222) | 13 ms | 15 ms | 8.57 | 8.50 | 0.13 |
| 4 | 744 | 3.87 | 280 (278-284) | 17 ms | 23 ms | 13.41 | 13.39 | 0.22 |
| 8 | 373 | 7.72 | 334 (329-337) | 29 ms | 42 ms | 22.68 | 22.37 | 0.50 |
| 16 | 187 | 15.40 | **353** (353-354) | 60 ms | 74 ms | 43.10 | 42.45 | 0.81 |
| 32 | 94 | 30.64 | 354 (352-362) | **130 ms** | 146 ms | 85.93 | 81.77 | 1.87 |

**Batching is worth 2.5x here, and it is finished by sixteen clients.** Those two
rows were re-run on their own at five repeats in both sweep directions, because
a three-repeat pass once put them 8% apart with the ranges not overlapping and
that would have been the wrong conclusion:

| | 16 clients | 32 clients |
|---|---|---|
| ascending | 350 (344-354) | 357 (351-370) |
| descending | 364 (347-365) | 363 (352-373) |

They overlap in both directions. Thirty-two buys nothing measurable over sixteen
and costs **1.9x the time to first token** (56-58 ms to 103-110 ms at p50).
Sixteen is where this server should be configured to stop.

**One client needs 2 880 forward passes for 2 880 tokens, sixteen need 187.**
Thirty times fewer trips through the model for two and a half times the
throughput, which is the whole shape of the thing: the trips are not free any
more.

Reproduce it:

```bash
go run ./cmd/braid -worker ./build/braid_worker.exe -model models/charlm
```

```bash
go run ./cmd/braidload -requests 96 -max-tokens 30 -repeat 3 -concurrency 1,2,4,8,16,32
```

---

## What the small model was hiding

This repository spent most of its life serving a model of **172 728**
parameters over a 64-id context, and almost everything it concluded from that
was about the harness rather than about serving.

| | 172 728 params, 64 ctx | 10 758 289 params, 256 ctx |
|---|---|---|
| what batching buys | **7.5x** | **2.5x** |
| where throughput saturates | 32 clients | **16 clients** |
| forward, batch 1 -> 12 | 0.77 -> ~1.0 ms | **6.81 -> 30.36 ms** |
| kernels per step | 0 to 60, discontinuous | **176, flat** |
| PCIe crossings per step | 5 each way below batch 6 | **1 each way, always** |
| the pipe, batch 32 | 0.68 ms, 12% of a step | **below the noise floor** |
| engine CUDA thresholds | two of them had to be moved | **both inert** |

Read the middle row first. The old forward went from 0.77 ms at a batch of one
to about 1.0 ms at twelve: **nearly flat**, because there was not enough
arithmetic in a 172 728-parameter model for the batch size to matter. Every
millisecond was launch overhead, transfer and dispatch. The new one goes from
6.81 to 30.36 over the same range - about 4.5 ms of fixed cost and 2.15 ms per
additional sequence - which is a GPU doing work.

Everything else in the table follows from that one fact:

**The 7.5x was mostly an artifact.** Batching amortises fixed cost. When almost
all of a step *is* fixed cost, batching looks spectacular. Give the GPU real
work and the honest number is 2.5x - still worth having, and no longer the
headline.

**Both engine thresholds are now inert**, and that took three merged pull
requests to discover was necessary and one bigger model to discover was
temporary. `ENGINE_CUDA_MIN_FLOPS` kept a batch of one entirely on the CPU:
zero kernels, at a threshold of 4.2 MFLOP against a feed-forward of 2.4. At 384
wide over 256 positions that same matmul is 302 MFLOP, which clears it seventy
times over. `ENGINE_CUDA_MIN_LAYERNORM` refused every LayerNorm below a batch of
six, forcing five PCIe crossings a step. Its input went from n x 6 144 elements
to n x 98 304, which clears the floor at a batch of one.

The measurement above was taken **with the engine's own defaults**, no flags:
176 kernels and one crossing each way at every batch size from one to twelve. No
discontinuity anywhere.

**None of that work was wrong** - the thresholds really were miscalibrated for
serving, the pull requests are still merged and still help a small model. What
was wrong was believing the numbers described inference rather than describing a
model too small to be inferred from. The fix was not a better measurement. It
was a bigger model.

### And the pipe stopped being measurable

The process boundary cost 0.68 ms at a batch of thirty-two, 12% of a step. Now
the subtraction that measures it - the wall clock minus what the worker reports
of itself - comes out **negative** at small batches: -0.03 ms at one, -0.21 ms
at eight. Two clocks disagreeing by a fraction of a millisecond over a step that
now takes eighty.

That is not a bug and it is not zero. It is the method running out of
resolution, and the code said so before it happened: the comment on `PipeMS`
promises the value is reported as measured rather than clamped, "because a
negative here means the measurement is wrong and that is worth seeing." At batch
thirty-two, where the frame is 32 KB, it is 2.24 ms and real again.


---

## The bug no number could have shown me

Every measurement in this README was taken while the model was being fed
garbage, and every measurement was correct.

The scheduler hands the backend a fixed 256-id window per sequence. A sequence
shorter than that has to be padded, and until 2026-08-26 the padding went at the
**front**: real ids at the end of the row, zeros before them. That is the
conventional choice, and with a causal mask it is wrong.

A causal mask hides the *future*. Position `i` may attend to `0..i` and nothing
beyond. It says nothing about padding, because padding is not a concept the mask
has — so the row being sampled, the last one, attended to every pad id in front
of it as though it were context. And id 0 in this alphabet is a **tab**.

A five-character prompt therefore reached the model as 251 tabs followed by
`func `, and the model did the only reasonable thing with that. Both samples
below are a single line, wrapped here to fit, with newlines and tabs shown as
escapes:

```
prompt: "func "
int64\n\t\t\t\t}\n\t\t\t\t\t\t\t\t\t\t\t\t}()\n\t\t\t\t\t\t})\n\t\t\t\t\t}()\n\t\t\t\t\tin =
append(io.channex, want)\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t
```

Same model, same weights, same seed, with the padding moved to the back and the
sampling told where the sequence actually ends:

```
prompt: "func "
index not of the output");\n    check_throws([&] { (void)A.slice(0, 0, 0); },
"positional_enconding throws");\n\n    // The one-thread engine` no is a decision
that is already the asserts a hang the\n
```

Still a 10 M-parameter character model trained for 24 minutes — it is not
*right*, it was never going to be right. But it is now answering the prompt.

The load harness sends `"the engine "`, eleven characters. **Every throughput
number on this page was measured with the model reading 245 tabs.** The numbers
are not wrong — the tensor is the same shape either way, the same 176 kernels
run, and the padding costs nothing it would not have cost as text. They were
simply never about generating anything.

**What makes this worth writing down is that nothing in the repository could
have caught it.** Twenty-eight tests, a measured divergence rate, percentiles, a
kernel counter, a chaos harness that kills processes under load — and a model
conditioned on 245 tabs satisfies every one of them. It is deterministic. It is
invariant to batch size. It fails over cleanly. Re-running the sweep above after
the fix gives the same counts to the digit — 2 880 forward passes at one client,
187 at sixteen, 176 kernels a step at every size — because the tensor is the same
shape either way and the padding costs exactly what text would have cost. The bug
lived in the one place none of the instruments pointed: whether the output
*meant* anything.

I found it by reading generated text for an unrelated reason.

The fix is the padding on the right, plus a per-row length in the frame so the
worker samples at position `length-1` instead of at the end of the window:

```
request   u32 magic 'B','R','D','6'      was BRD5
          u32 n
          i32 ids[n * kSeqLen]           right-padded now
          i32 length[n]                  new: real ids in each row
          f32 temperature[n]
          u64 seed[n]
```

The length cannot be inferred from the window — id 0 is a legal id — so it is a
field, and adding a field is what the version in the magic is for. Both sides
reject a length outside `1..kSeqLen` rather than clamping it: a length that
disagrees between the two ends means the frame is being misread, and sampling
some other position would answer the wrong question convincingly.

`TestWindowIsRightPadded` holds the scheduler to the new shape, including the
case that only exists now — the loop reuses one backing array across steps, so
with the padding at the back a short sequence lands on top of a longer one's ids
and the stale tail sits exactly where a backend that ignored `length` would read
it as context.

One consequence beyond correctness: a token's position no longer changes as the
sequence grows, up to the full 256. That is the precondition for a KV cache,
which is the next thing here — a cache keyed by position is worthless if every
step renumbers every token.

---

## The property that makes it correct, and how far it goes

Two claims live here and they are not equally strong. Keeping them apart is the
point of this section.

**The scheduler guarantees isolation.** Each sequence carries its own history,
its own temperature and its own seed, and the sampler is a fresh
`mt19937_64(seed)` per row. No row can read another's state, at any batch size,
on any hardware. That is structural.

**The arithmetic does not guarantee bit-identical logits.** A batch of *n* is a
different matrix product from a batch of one, the engine picks its matmul kernel
from the row count, and the sampler walks an inverse CDF — so a one-ULP
difference near a boundary flips a token and from there the sequences part.

So: **isolation is guaranteed, identity is measured.**

`TestBatchInvarianceDivergenceRate` measures it. The same window and the same
seed, sampled alone and then as one row of a batch of *n*, with the row's
position, its neighbours **and every row's length** redrawn every trial — so
what is measured is the batch rather than one arrangement of it, and rows that
sample at different positions are part of the arrangement.

| batch | trials | diverged |
|---|---|---|
| 2 | 5 000 | 0 |
| 4 | 5 000 | 0 |
| 8 | 5 000 | 0 |
| 16 | 5 000 | 0 |
| 32 | 5 000 | 0 |

Zero observations is not a rate of zero. By the rule of three, 0 in 25 000 puts
the 95% upper bound at **1.2 × 10⁻⁴** — about one flipped token in eight
thousand at worst, and possibly none at all. Raise `BRAID_DIVERGENCE_TRIALS` and
the bound tightens; the run above took eighteen minutes.

The test asserts a **ceiling of 1%**, not zero, deliberately. Nothing in the
engine promises the same reduction order at two batch sizes, so a test demanding
identity would assert a property the code does not have and would eventually
fail for being right. What it catches is the rate climbing.

`TestBatchingDoesNotChangeOutput` is the same property against the mock, whose
next token is a hash of the sequence's real ids and its length: bit-exact by
construction, and it catches every scheduler bug — a crossed row, a window built
from the wrong end, the wrong sequence advanced. It deliberately does *not* hash
the padding, for the same reason the model cannot see it: a mock that hashed the
whole row would go on passing if the length were ever threaded through wrong. Both tests also assert the sequences really did
share steps. An earlier version of the second passed while batching nothing at
all: the backend was fast enough that each request finished before the next
arrived, mean batch 1.00, invariant trivially satisfied, nothing proved.

---

## Killing a worker under load

`-workers N` puts a pool behind the scheduler. The scheduler does not know: a
`Pool` is a `Backend` like any other, and nothing in `internal/sched` changed to
support this.

That is not luck. **A worker holds no state between steps.** The sequence's
history lives in the scheduler and every step sends the whole 256-id window, so
there is no cache to rebuild and no session to migrate — a step that fails on
one worker is asked of the next, with the same bytes, and the caller cannot
tell.

128 generations of 40 tokens at sixteen clients, twice, against the same server:

| | requests | tokens/s | TTFT p50 | TTFT p95 | failed |
|---|---|---|---|---|---|
| nothing dies | 128 | 370 | 47 ms | 75 ms | 0 |
| one worker killed mid-run | 128 | 364 | 46 ms | 74 ms | **0** |

`Stop-Process -Force`, not a signal: a signal would exercise the shutdown path,
which is not the path in question. One death, one failover, one restart, and the
tail did not move.

**The blast radius of a worker dying is exactly one step**, because the
scheduler advances the batch one step at a time and that worker was holding
exactly one. This is a consequence of the design rather than a recovery
mechanism that had to be engineered: there was nothing to lose, so nothing was
lost. It is also the property a KV cache would spend, which is why the roadmap
says to measure the trade rather than assume it.

### What a pool is not

**It is not capacity.** The scheduler steps serially, so at any instant one
worker is computing and the rest are idle — three workers reach the same
throughput as one, which the table above shows by matching the single-worker
figure to within noise. The pool buys redundancy and nothing else on one card.

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
turns a hang into a death, which the pool already knows how to handle.

Reviewing that fix found the same bug twice more, one layer down. The reap after
the kill waited on its goroutine without a bound — correct exactly when the kill
lands, and the original hang again when it does not. And `Close` waited on
`cmd.Wait` without a bound, which matters because `retire()` closes from the
scheduler's own goroutine, and a worker wedged in a kernel call is not reading
its stdin to be asked to stop. Both are bounded now.

Three tests hold it, all in CI, none needing a GPU. The fake worker grew a
`hang` mode: read the frame, answer never, stay alive. Writing it took two
attempts — the first parked in `select{}`, which the Go runtime calls a deadlock
and answers by killing the process, producing the immediate EOF the mode existed
to avoid. The assertion caught it. A weaker one would have passed.

---

## Why the model runs in another process

`nvcc` on Windows compiles only through MSVC. cgo links only through a
GCC-compatible toolchain. A C++ static library from one is not linkable by the
other, so the engine cannot be linked into the Go binary at all.

A pipe has no ABI to disagree about. `braid_worker` is a C++ process that owns
the model and answers step requests over stdin and stdout in a fixed binary
frame; the Go side writes `n` windows and reads `n` ids back, plus the timings
and counters it needs to say where the step went.

It used to cost about 12% of a step. At this model's size
[it is below the noise floor](#and-the-pipe-stopped-being-measurable), and what
falls out of it is the shape the pool needs: a worker is a thing that can be
killed.

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
behind, which punished a caller for being briefly behind — HTTP clients are
bursty. Sizing the buffer to the most a request could ever produce removes the
question: a caller that reads nothing at all still completes, buffered, paying
for itself in memory and costing its batch neighbours nothing.

And one that is only visible in what a prompt costs: **only the last 256 ids of
a prompt are kept**, because `window()` takes the tail and the model has no other
context. Keeping the rest was a caller-controlled allocation with no purpose — at
a 1 MiB body limit and a queue of 256, up to a gigabyte of ids nothing would ever
read. Copied rather than resliced, since `history[len-n:]` would have fixed the
arithmetic and none of the memory.

---

## What the server says about itself

`/stats` reports the scheduler's counters, the step breakdown when the backend
can produce one, the pool when there is one, and **its own latency** as
percentiles over the last 1 024 completions.

That last part was missing for longer than it should have been. A comment in
`stats.go` said *"a server that computes its own averages is a server that hides
its tail"* — and the file it was written in exposed counters and a mean batch
size. The percentiles lived only in the load harness, which meant this server
could describe its own tail only while somebody was benchmarking it, which is
exactly when a tail matters least.

A window rather than a lifetime, because a p99 over every request since boot is
a number an hour of good service cannot move and an incident an hour ago never
leaves. Rejections are kept out of it: they have no time to first token, and
counting them as zero would drag every percentile down precisely when the server
is overloaded.

---

## How this was built

With heavy AI assistance, the same as
[cpp-ai-engine](https://github.com/DanielPastor05/cpp-ai-engine#how-this-was-built),
and it is said here for the same reason: a reader who works it out for themselves
has been told something the author would rather they had not noticed.

The design decisions are mine and they are the part worth reading — a stateless
worker so failover is a retry, one goroutine so the batch has no locking to get
wrong, a `Backend` seam narrow enough to test the scheduler with no model
present. So is the discipline the measurements are held to: every number on this
page was produced by running the thing, several of them contradicted what I
expected, and the ones that turned out wrong were corrected in place rather than
quietly replaced.

Four of those corrections are still on this page on purpose. A speedup credited
to the wrong change and fixed upstream in a second pull request. A 2.6× that was
forward-time only and smaller end to end. A 64× over-copy that reads like a
catastrophe in the source and measured at 8%. And the largest: a headline
speedup of 7.5× that turned out to be mostly a fact about a model too small to
serve. Leaving them visible is the only way the rest of the numbers mean
anything.

---

## Layout

```
cmd/braid/          the server
cmd/braidload/      the load harness that printed the tables
internal/sched/     the loop: admission, batching, cancellation, stats, latency
internal/backend/   the seam -- Backend is six methods; Mock, Worker and Pool implement them
internal/api/       HTTP and server-sent events
engine/             the C++ side: the model, the trainer, the worker process
third_party/        cpp-ai-engine, pinned as a submodule
```

`internal/sched` imports nothing that knows what a GPU is. The whole batching
argument is testable, and tested, with no model present: 28 tests run in CI on a
Linux box with no card in it, including the pool, the failover and the hang.

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
./build/braid_train . models/charlm 6000
```

Training is about 24 minutes on a 3060 Ti and lands at 0.97 bits/char. The
corpus is both repositories' own sources and documentation — a megabyte, which is
far too little for ten million parameters: the model memorises it rather than
learns from it, and what it writes is worth reading as evidence that the
machinery runs and nothing more. The serving measurements this checkpoint exists
for do not care what it learned, only how much arithmetic it takes to run.

The checkpoint is 43 MB and is not committed, because `braid_train` is seeded and
reproduces it.

Without `-worker`, the server runs a mock backend and says so on every startup,
because a server that quietly served plausible nonsense at good latencies is a
server whose numbers end up pasted somewhere.

---

## Next

Ordered by what a measurement says.

1. **A KV cache**, and publishing what it costs. The model recomputes the full
   256-id window every step, which is now most of the 6.6 ms a batch-1 step
   takes. Moving the padding to the right was the precondition and is done — a
   token's position is now fixed for the life of a sequence up to the full 256,
   and a cache keyed by position is worthless without that. Beyond 256 the window
   slides and every position changes again, which is the honest boundary of what
   a cache can be here. The interesting part is not the speedup: the stateless
   worker is what
   makes failover a retry with the same bytes, and a cache is state. The good
   answer is a cache that is not authoritative — the scheduler still holds the
   history, so a new worker can rebuild it with a prefill — and the deliverable
   is what that prefill costs, measured against the failover table above.
2. **Compare against a real serving system.** `cpp-ai-engine` opens with "1.70×
   slower than PyTorch, and that is the number worth publishing." This page
   compares against nothing. The same model and the same weights behind a minimal
   PyTorch server, same hardware and same harness, and publish the gap with the
   step breakdown that explains it.
3. **Memory as the scheduling resource.** With a cache at this context length,
   admission should count blocks rather than requests: a generation of 1 000
   tokens and one of 10 do not cost the same and `QueueDepth` pretends they do.
   That, and what eviction costs under pressure, is what PagedAttention is
   actually about.
4. **A device-side slice, in the engine.** The worker pulls the logits for every
   position and reads one row of each. It was 4–8% of a step at the old size and
   wants re-measuring at this one.
5. **Run the load generator off this machine.** Its goroutines share a CPU with
   the server and the worker, and that is the last uncontrolled variable in the
   table.

**Withdrawn: CUDA streams and compute/transfer overlap.** It was on this list and
it does not fit. Decoding is autoregressive — step *n+1* takes step *n*'s token
as input — so within one batch there is no independent work to overlap with. A
second stream needs a second batch, which needs a second card.

Also not here, deliberately: authentication, TLS, rate limiting, Kubernetes,
Prometheus. This serves a character model on a desk. The honest version is that
it should not be exposed to anything, and that is a sentence rather than a
feature.
