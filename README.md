# braid

A continuously batched inference server for
[cpp-ai-engine](https://github.com/DanielPastor05/cpp-ai-engine), written in Go.

Independent requests arrive whenever they arrive and are merged into a single
forward pass. A request that shows up mid-flight joins the batch at the next
step; a request that finishes leaves at the next step; **neither event changes
one character of what anybody else generates**, and there is a test that holds
the implementation to exactly that.

---

## Status

The scheduler, the HTTP surface and the load harness are done and measured. The
engine is not wired in yet — this runs against a mock backend, and **every
number on this page comes from that mock**, so every number on this page is a
statement about the scheduler and not about a model. The real ones arrive with
the CUDA backend.

| | |
|---|---|
| Scheduler, admission, backpressure, cancellation | done, 9 tests |
| HTTP + server-sent events | done |
| Load harness | done |
| CUDA backend over `cpp-ai-engine` | not started |
| Multi-worker router, worker death | not started |

---

## What batching buys, and where it stops paying

48 generations of 60 tokens at each concurrency level, against a mock whose step
costs a fixed 5 ms plus 200 µs per sequence in it. Same total work every row —
2 880 tokens — so the step count is directly comparable down the column.

| clients | steps | mean batch | tokens/s | TTFT p50 | TTFT p95 | TTFT p99 |
|---|---|---|---|---|---|---|
| 1 | 2 880 | 1.00 | 179 | 9 ms | 10 ms | 10 ms |
| 2 | 1 465 | 1.97 | 341 | 10 ms | 14 ms | 16 ms |
| 4 | 733 | 3.93 | 644 | 10 ms | 15 ms | 17 ms |
| 8 | 369 | 7.80 | 1 117 | 12 ms | 17 ms | 17 ms |
| 16 | 186 | 15.48 | 1 773 | 19 ms | 25 ms | 28 ms |
| 32 | 125 | 23.04 | 2 269 | 28 ms | 51 ms | 63 ms |
| 64 | 123 | 23.41 | 2 279 | 53 ms | **743 ms** | 753 ms |

**One client needs 2 880 forward passes to produce 2 880 tokens.** That is what
serving requests one at a time means, and the first row confirms the accounting
is honest: exactly one step per token, mean batch 1.00.

**Thirty-two clients need 125.** The same work, 23× fewer trips through the
model, and 12.7× the throughput.

**Sixty-four clients need 123, which is the interesting row.** Throughput moved
0.4% — from 2 269 to 2 279 tokens a second — and the p95 time to first token got
14.6× worse, from 51 ms to 743 ms. Past `-max-batch` the extra clients are not
being served faster, they are being queued, and the only thing more concurrency
buys is a longer wait before anyone hears anything back. A server that reported
only its throughput would call rows 6 and 7 equally good. They are not, and the
tail is where the difference lives.

Reproduce it:

```bash
go run ./cmd/braid -addr 127.0.0.1:8420 -mock-step 5ms
```

```bash
go run ./cmd/braidload -addr http://127.0.0.1:8420 -requests 48 -max-tokens 60
```

---

## The property that makes it correct

Continuous batching is only worth anything if a sequence cannot tell it happened.
The model's window is 64 ids wide, so a batch of *n* sequences is one `(n, 64)`
tensor, and a scheduler bug — a window padded from the wrong end, two rows
crossed, the wrong sequence advanced — produces text that is subtly wrong rather
than an error that is obviously wrong. Nothing crashes. The output is just not
what that request asked for.

So it is tested directly, in `TestBatchingDoesNotChangeOutput`: run a request
alone and keep the text; run the identical request again while eight neighbours
of assorted lengths join and leave around it, on a backend with different
timings; assert the text is identical, character for character. The mock's next
token is a hash of the entire window, so any crossed row changes it.

The test also asserts that the sequences really did share steps. An earlier
version passed while batching nothing at all — the backend was fast enough that
each request finished before the next arrived, mean batch 1.00, invariant
trivially satisfied, nothing proved. Now it fails if the mean batch never
exceeds one, which is the check that keeps the test from quietly becoming
decorative.

---

## Three decisions worth the words

**The queue rejects rather than grows.** Past `-queue` the answer is HTTP 429
with a `Retry-After`. An unbounded queue does not make a server faster, it
converts a throughput problem into a latency problem and reports neither.

**Deadlines are checked at admission, not at completion.** A caller that sets
`max_wait_ms` and waits longer than that is rejected *before* the first forward
pass. Rejecting early costs nothing; discovering it after the GPU has already
paid for the work wastes the work.

**A stream's buffer is exactly `max_tokens` long, so the loop's send can never
block.** The first version capped it at 32 tokens and killed whoever fell
behind. That punished a caller for being briefly behind — HTTP clients are
bursty — and it meant one slow reader could take out its own request for no good
reason. Sizing the buffer to the most the request could ever produce removes the
question: a caller that reads nothing at all still completes, buffered, paying
for itself in memory and costing its batch neighbours nothing. `MaxTokensLimit`
is what bounds the memory, and it is also what bounds how long one sequence can
hold a slot in the batch — the same number doing both jobs.

---

## Layout

```
cmd/braid/          the server
cmd/braidload/      the load harness that printed the table
internal/sched/     the loop: admission, batching, cancellation, stats
internal/backend/   the seam -- Backend is four methods, Mock implements them
internal/api/       HTTP and server-sent events
```

`internal/sched` does not import anything that knows what a GPU is. The whole
batching argument is testable, and tested, with no model present; the CUDA
backend has to satisfy `Backend` and nothing else.

---

## Next

1. **The CUDA backend.** `Backend.Step` over `cpp-ai-engine`'s character model
   via cgo, so an `(n, 64)` batch is one real forward pass. Then the table above
   gets run again and the mock numbers come off this page.
2. **A KV cache.** The engine recomputes the full 64-id window every step, which
   is the honest baseline to measure a cache against.
3. **A router and more than one worker**, then `kill -9` one under load and
   publish the recovery curve.
