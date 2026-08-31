package sched

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/DanielPastor05/braid/internal/backend"
)

// fastMock is the mock with its sleeps removed, for tests about logic rather
// than about timing.
func fastMock() *backend.Mock {
	m := backend.NewMock()
	m.Base = 0
	m.PerSeq = 0
	return m
}

// drain reads a stream to the end and returns the text and the result.
func drain(t *testing.T, tokens <-chan Token, done <-chan Result) (string, Result) {
	t.Helper()
	var sb strings.Builder
	for tok := range tokens {
		sb.WriteString(tok.Text)
	}
	select {
	case res := <-done:
		return sb.String(), res
	case <-time.After(5 * time.Second):
		t.Fatal("result never arrived after the token channel closed")
		return "", Result{}
	}
}

func run(t *testing.T, s *Scheduler, req Request) (string, Result) {
	t.Helper()
	tokens, done, err := s.Submit(context.Background(), req)
	if err != nil {
		t.Fatalf("submit: %v", err)
	}
	return drain(t, tokens, done)
}

// TestBatchingDoesNotChangeOutput is the property the whole package exists to
// provide: a sequence's output depends on the sequence, and on nothing about
// who else happened to be in the batch when it ran.
//
// It is checked the only way it can be checked: by running the same request
// alone, then running it again while other requests join and leave around it,
// and comparing the text. Any bug that crosses two rows of the batch, pads a
// window wrong, or advances the wrong sequence's history shows up here as
// different text, because the mock's next token is a hash of the whole window.
func TestBatchingDoesNotChangeOutput(t *testing.T) {
	subject := Request{Prompt: "the engine ", MaxTokens: 40, Temperature: 0.8, Seed: 7}

	alone, err := New(fastMock(), Config{MaxBatch: 32, QueueDepth: 64, MaxTokensLimit: 1024})
	if err != nil {
		t.Fatal(err)
	}
	want, res := run(t, alone, subject)
	if res.Err != nil {
		t.Fatalf("the run on its own failed: %v", res.Err)
	}
	if err := alone.Close(); err != nil {
		t.Fatal(err)
	}
	if len(want) != subject.MaxTokens {
		t.Fatalf("expected %d characters from a run on its own, got %d", subject.MaxTokens, len(want))
	}

	// Now the same request, with company: eight neighbours of assorted lengths,
	// submitted at staggered moments so that they join the batch on different
	// steps and leave it on different steps.
	//
	// This backend is deliberately slower than the one above. A step that costs
	// nothing finishes each request before the next one arrives, and the batch
	// never exceeds one -- which is a true result about an idle server and no
	// test of anything. A millisecond a step is enough for the arrivals to
	// overlap. The timings differ between the two runs on purpose: the text must
	// not depend on them either.
	slow := backend.NewMock()
	slow.Base = time.Millisecond
	slow.PerSeq = 0

	crowded, err := New(slow, Config{MaxBatch: 32, QueueDepth: 64, MaxTokensLimit: 1024})
	if err != nil {
		t.Fatal(err)
	}
	defer crowded.Close()

	var wg sync.WaitGroup
	for i := range 8 {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			time.Sleep(time.Duration(i) * 700 * time.Microsecond)
			run(t, crowded, Request{
				Prompt:      "noise",
				MaxTokens:   5 + i*7,
				Temperature: 0.5 + float32(i)/10,
				Seed:        uint64(1000 + i),
			})
		}(i)
	}

	got, res := run(t, crowded, subject)
	wg.Wait()

	if res.Err != nil {
		t.Fatalf("the run in company failed: %v", res.Err)
	}
	if got != want {
		t.Errorf("sharing a batch changed the output.\n alone: %q\nshared: %q", want, got)
	}

	// And the test is only meaningful if the sequences really did share steps.
	if snap := crowded.Stats(); snap.MeanBatch <= 1.0 {
		t.Errorf("mean batch was %.2f, so nothing was ever batched and this test proved nothing",
			snap.MeanBatch)
	}
}

// TestMeanBatchRisesWithConcurrency is the other half: the loop must actually
// merge concurrent work, not merely be capable of it.
func TestMeanBatchRisesWithConcurrency(t *testing.T) {
	m := backend.NewMock()
	m.Base = 2 * time.Millisecond // slow enough that arrivals pile up
	m.PerSeq = 0

	s, err := New(m, Config{MaxBatch: 32, QueueDepth: 128, MaxTokensLimit: 1024})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	var wg sync.WaitGroup
	for i := range 16 {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			run(t, s, Request{Prompt: "x", MaxTokens: 20, Temperature: 1, Seed: uint64(i)})
		}(i)
	}
	wg.Wait()

	snap := s.Stats()
	if snap.Completed != 16 {
		t.Fatalf("expected 16 completed, got %d (failed %d)", snap.Completed, snap.Failed)
	}
	// Sixteen clients of twenty tokens each is 320 tokens. Unbatched that is
	// 320 steps; perfectly batched it is 20. Anything near the floor means the
	// loop is admitting one at a time.
	if snap.MeanBatch < 4 {
		t.Errorf("mean batch %.2f over %d steps: sixteen concurrent clients were barely merged",
			snap.MeanBatch, snap.Steps)
	}
	if snap.Steps >= 320 {
		t.Errorf("%d steps for 320 tokens is one step per token, which is what not batching looks like",
			snap.Steps)
	}
}

func TestQueueFullIsRejectedNotQueued(t *testing.T) {
	m := backend.NewMock()
	m.Base = 50 * time.Millisecond // hold the loop so the queue fills
	m.PerSeq = 0

	s, err := New(m, Config{MaxBatch: 1, QueueDepth: 2, MaxTokensLimit: 1024})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	var rejected int
	for range 20 {
		_, _, err := s.Submit(context.Background(), Request{
			Prompt: "x", MaxTokens: 50, Temperature: 1,
		})
		if errors.Is(err, ErrQueueFull) {
			rejected++
		}
	}
	if rejected == 0 {
		t.Fatal("a depth-2 queue accepted twenty requests without rejecting one")
	}
	if snap := s.Stats(); snap.Rejected != int64(rejected) {
		t.Errorf("counted %d rejections, stats reported %d", rejected, snap.Rejected)
	}
}

func TestDeadlineIsCheckedBeforeTheGPUIsUsed(t *testing.T) {
	m := backend.NewMock()
	m.Base = 30 * time.Millisecond
	m.PerSeq = 0

	s, err := New(m, Config{MaxBatch: 1, QueueDepth: 64, MaxTokensLimit: 1024})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	// One long request to occupy the loop, then one that will not wait.
	_, _, err = s.Submit(context.Background(), Request{Prompt: "x", MaxTokens: 30, Temperature: 1})
	if err != nil {
		t.Fatal(err)
	}

	tokens, done, err := s.Submit(context.Background(), Request{
		Prompt: "y", MaxTokens: 10, Temperature: 1, MaxWait: time.Millisecond,
	})
	if err != nil {
		t.Fatalf("the request should be rejected at admission, not at submission: %v", err)
	}
	text, res := drain(t, tokens, done)

	if !errors.Is(res.Err, ErrDeadlineExceeded) {
		t.Fatalf("expected ErrDeadlineExceeded, got %v", res.Err)
	}
	if text != "" {
		t.Errorf("a request past its deadline generated %q; the work should never have started", text)
	}
	if snap := s.Stats(); snap.Expired != 1 {
		t.Errorf("expected one expired request in stats, got %d", snap.Expired)
	}
}

func TestCancellationStopsGeneration(t *testing.T) {
	m := backend.NewMock()
	m.Base = 2 * time.Millisecond
	m.PerSeq = 0

	s, err := New(m, Config{MaxBatch: 8, QueueDepth: 32, MaxTokensLimit: 1024})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	ctx, cancel := context.WithCancel(context.Background())
	tokens, done, err := s.Submit(ctx, Request{Prompt: "x", MaxTokens: 1000, Temperature: 1})
	if err != nil {
		t.Fatal(err)
	}

	for range 3 {
		if _, ok := <-tokens; !ok {
			t.Fatal("the stream closed before three tokens arrived")
		}
	}
	cancel()

	for range tokens { //nolint:revive // draining what was already buffered
	}
	res := <-done
	if !errors.Is(res.Err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", res.Err)
	}
	// At two milliseconds a token, three tokens read and then a cancel is a
	// handful of steps. The ceiling is loose because this is a timing test on a
	// shared machine, but it is far enough below MaxTokens to distinguish
	// "stopped when asked" from "ran to completion regardless".
	if res.Generated > 50 {
		t.Errorf("generated %d tokens before noticing the cancel; it should stop within a step or two",
			res.Generated)
	}
}

// TestStalledCallerDoesNotStallTheBatch checks the invariant that lets the loop
// send tokens without ever selecting on the send: a caller that never reads a
// single token still runs to completion, buffered, and the sequences sharing
// its batch neither wait for it nor fail because of it.
//
// An earlier version of this package bounded the buffer at 32 tokens and killed
// whoever overran it. That punished a caller for being briefly behind, and this
// test is what the design became instead.
func TestStalledCallerDoesNotStallTheBatch(t *testing.T) {
	s, err := New(fastMock(), Config{MaxBatch: 8, QueueDepth: 32, MaxTokensLimit: 1024})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	// Submitted and then deliberately never read until the very end.
	stalledTokens, stalledDone, err := s.Submit(context.Background(), Request{
		Prompt: "stalled", MaxTokens: 1000, Temperature: 1,
	})
	if err != nil {
		t.Fatal(err)
	}

	// A neighbour that does read must finish, unaffected.
	text, res := run(t, s, Request{Prompt: "polite", MaxTokens: 50, Temperature: 1, Seed: 3})
	if res.Err != nil {
		t.Fatalf("the reading caller was punished for its neighbour: %v", res.Err)
	}
	if len(text) != 50 {
		t.Fatalf("expected 50 characters, got %d", len(text))
	}

	select {
	case stalled := <-stalledDone:
		if stalled.Err != nil {
			t.Errorf("the caller that read nothing was failed for it: %v", stalled.Err)
		}
		if stalled.Generated != 1000 {
			t.Errorf("expected all 1000 tokens buffered, got %d", stalled.Generated)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the stalled request never finished, which means the loop blocked on its send")
	}

	// And every one of those tokens is still there to be collected.
	var late int
	for range stalledTokens {
		late++
	}
	if late != 1000 {
		t.Errorf("expected 1000 buffered tokens after the fact, got %d", late)
	}
}

// TestWindowIsRightPadded holds the window to the shape the causal mask makes
// safe, and to the length that says where the real ids stop.
//
// It was left-padded until 2026-08-26, which was wrong in a way nothing caught:
// the mask hides the future and not the padding, so the position being sampled
// attended to every pad id in front of it. Id 0 in the model's alphabet is a
// tab, so a five-character prompt reached the model as 251 tabs and the model,
// correctly, answered with tabs. Every test in this package went on passing --
// the output was deterministic, invariant to batch size and the right shape, and
// none of those properties has anything to say about whether it means
// something.
func TestWindowIsRightPadded(t *testing.T) {
	seq := &sequence{history: []int32{7, 8, 9}}
	got := make([]int32, 5)

	if n := seq.window(got); n != 3 {
		t.Errorf("a history of 3 reported length %d", n)
	}
	want := []int32{7, 8, 9, 0, 0}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("short history: got %v, want %v", got, want)
		}
	}

	// A history longer than the window keeps the tail, because the model's
	// context is the most recent ids and not the first ones. At the full width
	// there is no padding and the length is the width.
	seq = &sequence{history: []int32{1, 2, 3, 4, 5, 6, 7}}
	if n := seq.window(got); n != 5 {
		t.Errorf("a full window reported length %d", n)
	}
	want = []int32{3, 4, 5, 6, 7}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("long history: got %v, want %v", got, want)
		}
	}

	// The loop reuses one backing array across steps, and with the padding at
	// the back it is exactly where a previous longer sequence's ids would still
	// be sitting. A backend that ignored the length would read them as context.
	seq = &sequence{history: []int32{4}}
	if n := seq.window(got); n != 1 {
		t.Errorf("a history of 1 reported length %d", n)
	}
	want = []int32{4, 0, 0, 0, 0}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("reused buffer: got %v, want %v -- the last sequence is still in it",
				got, want)
		}
	}

	// An empty prompt has no last real position to sample, so it borrows the
	// pad id and generates what follows it. Length 0 would sample at index -1.
	seq = &sequence{}
	if n := seq.window(got); n != 1 {
		t.Errorf("an empty history reported length %d, which is not a position", n)
	}
}

// TestAlignmentAndWidthAreCounted holds the two numbers that say what a
// position-keyed key/value cache could do here.
//
// A batch computes every row to the width of its longest, so the width is what
// a step costs in positions. And a cache keyed by position shares one write
// offset across the batch, so it can only serve a step whose rows are all at the
// same position -- which continuous batching does not arrange, because requests
// arrive when they arrive. That is the measurement, not an opinion, and it is
// the reason the cache landed in the engine and not in the worker.
func TestAlignmentAndWidthAreCounted(t *testing.T) {
	s, err := New(fastMock(), Config{MaxBatch: 8, QueueDepth: 32, MaxTokensLimit: 64})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	// One request on its own: every step has exactly one row, so every step is
	// trivially aligned.
	tokens, done, err := s.Submit(context.Background(),
		Request{Prompt: "abc", MaxTokens: 5, Temperature: 0.7, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	if _, res := drain(t, tokens, done); res.Err != nil {
		t.Fatal(res.Err)
	}

	snap := s.Stats()
	if snap.AlignedShare != 1 {
		t.Errorf("a single sequence gave an aligned share of %v, want 1", snap.AlignedShare)
	}
	// Three prompt ids, then a token per step: widths 3, 4, 5, 6, 7 -> mean 5.
	if snap.MeanWidth < 3 || snap.MeanWidth > 7 {
		t.Errorf("mean width %v is outside the range the sequence spans", snap.MeanWidth)
	}
}

func TestSubmitRejectsNonsense(t *testing.T) {
	s, err := New(fastMock(), Default())
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	for _, req := range []Request{
		{Prompt: "x", MaxTokens: 0, Temperature: 1},
		{Prompt: "x", MaxTokens: 10, Temperature: 0},
		{Prompt: "x", MaxTokens: 10, Temperature: -1},
	} {
		if _, _, err := s.Submit(context.Background(), req); err == nil {
			t.Errorf("accepted %+v", req)
		}
	}
}

// TestTheServerReportsItsOwnTail covers what the load harness cannot: a server
// that can only describe its latency while somebody is benchmarking it is a
// server with no latency observability at all.
func TestTheServerReportsItsOwnTail(t *testing.T) {
	m := backend.NewMock()
	m.Base = time.Millisecond
	m.PerSeq = 0

	s, err := New(m, Config{MaxBatch: 4, QueueDepth: 32, MaxTokensLimit: 64})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	if empty := s.Stats().Latency; empty.Samples != 0 {
		t.Errorf("a server that has served nothing reported %d samples", empty.Samples)
	}

	for i := range 20 {
		run(t, s, Request{Prompt: "x", MaxTokens: 5, Temperature: 1, Seed: uint64(i)})
	}

	got := s.Stats().Latency
	if got.Samples != 20 {
		t.Fatalf("twenty requests produced %d samples", got.Samples)
	}
	// Percentiles have to be ordered, non-zero, and inside the total. A p99 that
	// came back as zero would look like a very fast server.
	if !(got.TTFTP50MS > 0 && got.TTFTP50MS <= got.TTFTP95MS && got.TTFTP95MS <= got.TTFTP99MS) {
		t.Errorf("time to first token is not a distribution: p50 %.3f, p95 %.3f, p99 %.3f",
			got.TTFTP50MS, got.TTFTP95MS, got.TTFTP99MS)
	}
	if got.TotalP50MS < got.TTFTP50MS {
		t.Errorf("a request finished before its first token: total p50 %.3f, ttft p50 %.3f",
			got.TotalP50MS, got.TTFTP50MS)
	}
}

// TestRejectionsStayOutOfTheLatencyWindow guards the number against the mistake
// that would flatter it most. A rejected request has no time to first token,
// and counting it as zero would drag every percentile down exactly when the
// server is overloaded and the tail is the only thing worth looking at.
func TestRejectionsStayOutOfTheLatencyWindow(t *testing.T) {
	m := backend.NewMock()
	m.Base = 20 * time.Millisecond
	m.PerSeq = 0

	s, err := New(m, Config{MaxBatch: 1, QueueDepth: 1, MaxTokensLimit: 128})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	var rejected int
	for range 30 {
		if _, _, err := s.Submit(context.Background(), Request{
			Prompt: "x", MaxTokens: 40, Temperature: 1,
		}); errors.Is(err, ErrQueueFull) {
			rejected++
		}
	}
	if rejected == 0 {
		t.Fatal("nothing was rejected, so this test measured nothing")
	}

	if got := s.Stats().Latency; got.Samples > 0 && got.TTFTP50MS == 0 {
		t.Error("a percentile of zero: rejections are being counted as instant successes")
	}
}

// TestEverySequenceGetsACacheSlot is an invariant, and writing it is what
// turned a metric I was about to ship into one I understood.
//
// It was going to be a degradation curve: slots run out, the sequences that go
// without are served by recomputing rather than refused, and the ratio says how
// much slower the server got. That is a good failure mode and it is not this
// server's, because it cannot happen here.
//
// There are MaxBatch slots. A sequence takes one in begin(), begin() is only
// reached while len(active) < MaxBatch, and every active sequence holds exactly
// one until finish() gives it back. So live slots equal len(active), which is
// below MaxBatch, so the free list is never empty. The -1 branch is a safety
// net over an invariant rather than a path load can reach.
//
// The counter stays, and what it is for is this: if it ever moves, the
// invariant above has broken and a sequence is quietly running slower than it
// should. Zero is the assertion.
func TestEverySequenceGetsACacheSlot(t *testing.T) {
	const slots = 3
	const clients = 8

	s, err := New(fastMock(), Config{MaxBatch: slots, QueueDepth: 64, MaxTokensLimit: 64})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	var wg sync.WaitGroup
	texts := make([]string, clients)
	for i := range clients {
		wg.Add(1)
		go func() {
			defer wg.Done()
			text, res := run(t, s, Request{
				Prompt: "slots", MaxTokens: 12, Temperature: 0.7, Seed: uint64(i),
			})
			if res.Err != nil {
				t.Errorf("client %d: %v", i, res.Err)
				return
			}
			texts[i] = text
		}()
	}
	wg.Wait()

	for i, text := range texts {
		if len(text) == 0 {
			t.Errorf("client %d generated nothing", i)
		}
	}

	snap := s.Stats()
	if snap.Uncached != 0 {
		t.Errorf("%d sequences ran without a cache slot; with MaxBatch slots and at "+
			"most MaxBatch active, that should be impossible", snap.Uncached)
	}
	if snap.Cached != int64(clients) {
		t.Errorf("%d clients were served but %d were counted as taking a slot",
			clients, snap.Cached)
	}

	// And the slots come back. A leak here would not fail anything above -- the
	// free list would just drain and later sequences would run uncached, which
	// is slower and correct and would go unnoticed without this.
	if len(s.freeSlots) != slots {
		t.Errorf("%d of %d slots came back after everything finished",
			len(s.freeSlots), slots)
	}
}

// TestADeadlineIsRefusedAtTheDoorRatherThanInTheQueue is the admission the
// open-loop measurement asked for.
//
// Offered more than it can serve, this server used to admit everything: the
// queue is 256 deep, it swallowed the whole overload, and every request was
// answered late without one of them being told. Queue depth answers "is there
// room", and a caller with a deadline asked "will I be served in time".
//
// So a request that says what late means to it is refused at submission when the
// queue ahead of it is longer than that. The two halves both matter and the
// second is the one that would go unnoticed: a server that refuses eagerly is
// worse than one that queues, so an empty queue must still admit.
func TestADeadlineIsRefusedAtTheDoorRatherThanInTheQueue(t *testing.T) {
	m := backend.NewMock()
	m.Base = 5 * time.Millisecond // slow enough that a queue means something
	m.PerSeq = 0

	s, err := New(m, Config{MaxBatch: 2, QueueDepth: 64, MaxTokensLimit: 256})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	// An empty queue predicts no wait, so a deadline of any size is admitted.
	// This is the half that catches an over-eager refusal.
	tokens, done, err := s.Submit(t.Context(), Request{
		Prompt: "first", MaxTokens: 4, Temperature: 0.7, MaxWait: time.Millisecond,
	})
	if err != nil {
		t.Fatalf("the first request was refused against an empty queue: %v", err)
	}
	drain(t, tokens, done)

	// Now fill the queue behind a slow batch, with requests that have no
	// deadline of their own so none of them is refused on the way in.
	var wg sync.WaitGroup
	for i := range 40 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			tok, res, err := s.Submit(t.Context(), Request{
				Prompt: "filler", MaxTokens: 64, Temperature: 0.7, Seed: uint64(i),
			})
			if err != nil {
				return // ErrQueueFull is fine here; the point is the queue is deep
			}
			drain(t, tok, res)
		}()
	}

	// Give the queue time to build and the scheduler time to measure a step, so
	// predictedWait has both of its inputs.
	deadline := time.Now().Add(2 * time.Second)
	var refused error
	for time.Now().Before(deadline) {
		_, _, err := s.Submit(t.Context(), Request{
			Prompt: "impatient", MaxTokens: 64, Temperature: 0.7, MaxWait: time.Millisecond,
		})
		if errors.Is(err, ErrWouldMissDeadline) {
			refused = err
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if refused == nil {
		t.Error("a one-millisecond deadline was never refused behind a full queue of " +
			"64-token generations, which is the case this exists for")
	}

	wg.Wait()
}
