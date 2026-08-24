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

func TestWindowIsLeftPadded(t *testing.T) {
	seq := &sequence{history: []int32{7, 8, 9}}
	got := make([]int32, 5)
	seq.window(got)
	want := []int32{0, 0, 7, 8, 9}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("short history: got %v, want %v", got, want)
		}
	}

	// A history longer than the window keeps the tail, because the model's
	// context is the most recent ids and not the first ones.
	seq = &sequence{history: []int32{1, 2, 3, 4, 5, 6, 7}}
	seq.window(got)
	want = []int32{3, 4, 5, 6, 7}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("long history: got %v, want %v", got, want)
		}
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
