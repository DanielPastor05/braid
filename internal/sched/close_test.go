package sched

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/DanielPastor05/braid/internal/backend"
)

// TestCloseAnswersRequestsStillInTheQueue holds Close to what its doc says.
//
// It used to fail everything the *loop* was holding and nothing else. The loop
// returns on stop without draining the queue, so a request that had been
// accepted but not yet admitted never had its stream closed and never received
// a Result: a caller ranging over the token channel waited for ever. Nothing in
// the server hit it -- main stops the listener first, so the queue drains while
// the handlers do, and the process exits straight after -- which is exactly why
// only a test was ever going to find it.
func TestCloseAnswersRequestsStillInTheQueue(t *testing.T) {
	slow := backend.NewMock()
	slow.Base = 200 * time.Millisecond
	slow.PerSeq = 0

	// One row at a time, so everything after the first waits in the queue.
	s, err := New(slow, Config{MaxBatch: 1, QueueDepth: 64, MaxTokensLimit: 1024})
	if err != nil {
		t.Fatal(err)
	}

	const requests = 6
	dones := make([]<-chan Result, 0, requests)
	for i := range requests {
		_, done, err := s.Submit(context.Background(),
			Request{Prompt: "x", MaxTokens: 50, Temperature: 1})
		if err != nil {
			t.Fatalf("submit %d: %v", i, err)
		}
		dones = append(dones, done)
	}
	// Long enough for one to be admitted and the rest to be sitting in the queue.
	time.Sleep(50 * time.Millisecond)

	go s.Close()

	for i, done := range dones {
		select {
		case res := <-done:
			// The error is not the assertion -- the one sequence that was
			// already running may finish normally. Receiving at all is.
			_ = res
		case <-time.After(5 * time.Second):
			t.Fatalf("request %d never received a Result after Close: its caller would wait for ever", i)
		}
	}
}

// TestSubmitAcceptedAfterCloseIsAnswered covers the window the drain alone does
// not: a Submit that passes its check of stop while Close is running, and sends
// into a queue nobody will read again. It is a race, so it is run many times and
// under -race in CI rather than reasoned about.
func TestSubmitAcceptedAfterCloseIsAnswered(t *testing.T) {
	for attempt := range 200 {
		s, err := New(fastMock(), Config{MaxBatch: 1, QueueDepth: 64, MaxTokensLimit: 1024})
		if err != nil {
			t.Fatal(err)
		}

		var wg sync.WaitGroup
		wg.Add(1)
		go func() {
			defer wg.Done()
			s.Close()
		}()

		_, done, err := s.Submit(context.Background(),
			Request{Prompt: "x", MaxTokens: 8, Temperature: 1})
		if err != nil {
			// Told no, which is the other correct answer.
			wg.Wait()
			continue
		}
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Fatalf("attempt %d: a request accepted around Close was never answered", attempt)
		}
		wg.Wait()
	}
}

// TestTheFreeSlotGaugeTracksTheList is the metric that would have caught the
// slot bug, and the reason it exists is that the two counters beside it could
// not: they notice a free list that shrinks, and that one grew.
func TestTheFreeSlotGaugeTracksTheList(t *testing.T) {
	const slots = 4
	// Slow, because the token channel is buffered to MaxTokens: with a fast
	// backend the loop generates every token and finishes the sequence whether
	// or not anyone is reading, so holding the reader would not hold a slot.
	slow := backend.NewMock()
	slow.Base = 20 * time.Millisecond
	slow.PerSeq = 0
	s, err := New(slow, Config{
		MaxBatch: slots, QueueDepth: 32, MaxTokensLimit: 64, CacheSlots: slots,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	if snap := s.Stats(); snap.FreeSlots != slots || snap.CacheSlots != slots {
		t.Fatalf("at rest the gauge reads %d free of %d, want %d of %d",
			snap.FreeSlots, snap.CacheSlots, slots, slots)
	}

	// In flight, so the gauge has to follow takeSlot and not only releaseSlot.
	// Read through Stats rather than off the list: the loop owns the list, and
	// the whole reason this counter exists is that reading it from here races.
	held := make(chan struct{})
	var wg sync.WaitGroup
	for range 3 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			tokens, done, err := s.Submit(context.Background(),
				Request{Prompt: "x", MaxTokens: 40, Temperature: 1})
			if err != nil {
				return
			}
			<-tokens // one token in hand means this sequence is admitted
			<-held
			for range tokens {
			}
			<-done
		}()
	}
	deadline := time.Now().Add(2 * time.Second)
	for {
		if snap := s.Stats(); snap.FreeSlots == slots-3 {
			break
		} else if time.Now().After(deadline) {
			t.Fatalf("three sequences are in flight and the gauge still reads %d free of %d",
				snap.FreeSlots, snap.CacheSlots)
		}
		time.Sleep(time.Millisecond)
	}
	close(held)
	wg.Wait()

	// Cancelled in the queue, which is the shape that used to invent slots.
	for i := range 5 {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_, done, err := s.Submit(ctx, Request{Prompt: "x", MaxTokens: 4, Temperature: 1})
		if err != nil {
			t.Fatalf("submit %d: %v", i, err)
		}
		<-done
	}
	time.Sleep(50 * time.Millisecond)

	snap := s.Stats()
	if snap.FreeSlots > snap.CacheSlots {
		t.Fatalf("the gauge reads %d slots free of %d that exist, which is the bug it is here to show",
			snap.FreeSlots, snap.CacheSlots)
	}
	if snap.FreeSlots != slots {
		t.Fatalf("the gauge reads %d free, want %d: nothing should be held", snap.FreeSlots, slots)
	}
	if int(snap.FreeSlots) != len(s.freeSlots) {
		t.Fatalf("the gauge says %d and the list holds %d", snap.FreeSlots, len(s.freeSlots))
	}
}
