package sched

import (
	"context"
	"testing"
	"time"
)

// TestTheFreeListNeverHoldsASlotTwice is the invariant the cache depends on,
// and it was false for every request that died in the queue.
//
// A slot is a row of the worker's key/value cache. Two live sequences holding
// the same row is not a slowdown, it is wrong output: the worker decides a slot
// is current by comparing filled[slot] against length-1, and it has no notion of
// whose row it is. Two sequences at the same length both skip the refill and
// read each other's keys. At different lengths they merely thrash.
//
// The bug was that `sequence.slot` began at the zero value, and zero is a slot.
// begin() calls finish() before takeSlot() on both rejection paths -- a
// cancelled context and a missed MaxWait -- finish() releases, and releaseSlot()
// only ignores negatives. So a request cancelled while queued handed back slot 0
// having never held it. api.go passes r.Context(), which Go cancels when a
// client disconnects, so an ordinary abandoned request was enough. Five of them
// left the free list holding [3 2 1 0 0 0 0 0 0].
//
// Counting is not the assertion. A test that only checked the length would pass
// on a list of nine distinct slots, and the failure that matters is duplication.
func TestTheFreeListNeverHoldsASlotTwice(t *testing.T) {
	const slots = 4

	check := func(t *testing.T, s *Scheduler, after string) {
		t.Helper()
		seen := map[int32]bool{}
		for _, slot := range s.freeSlots {
			if seen[slot] {
				t.Fatalf("after %s the free list holds slot %d twice: %v", after, slot, s.freeSlots)
			}
			seen[slot] = true
		}
		if len(s.freeSlots) > slots {
			t.Fatalf("after %s the free list holds %d slots, more than the %d that exist: %v",
				after, len(s.freeSlots), slots, s.freeSlots)
		}
	}

	t.Run("cancelled while queued", func(t *testing.T) {
		s, err := New(fastMock(), Config{
			MaxBatch: slots, QueueDepth: 64, MaxTokensLimit: 1024, CacheSlots: slots,
		})
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()

		for i := range 5 {
			ctx, cancel := context.WithCancel(context.Background())
			cancel() // dead before the loop can ever admit it
			_, done, err := s.Submit(ctx, Request{Prompt: "x", MaxTokens: 4, Temperature: 1})
			if err != nil {
				t.Fatalf("submit %d: %v", i, err)
			}
			<-done
		}
		check(t, s, "five requests cancelled in the queue")
	})

	// The same shape through the other rejection path, which shares the bug and
	// would not be covered by testing cancellation alone.
	t.Run("MaxWait missed in the queue", func(t *testing.T) {
		s, err := New(fastMock(), Config{
			MaxBatch: slots, QueueDepth: 64, MaxTokensLimit: 1024, CacheSlots: slots,
		})
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()

		for i := range 5 {
			_, done, err := s.Submit(context.Background(), Request{
				Prompt: "x", MaxTokens: 4, Temperature: 1,
				// Already missed by the time begin() looks at it.
				MaxWait: time.Nanosecond,
			})
			if err != nil {
				t.Fatalf("submit %d: %v", i, err)
			}
			<-done
		}
		check(t, s, "five requests that missed their MaxWait")
	})

	// And the ordinary path, so the test also says that slots come back at all
	// rather than only that they are not invented.
	t.Run("slots come back when sequences finish", func(t *testing.T) {
		s, err := New(fastMock(), Config{
			MaxBatch: slots, QueueDepth: 64, MaxTokensLimit: 1024, CacheSlots: slots,
		})
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()

		for i := range 8 {
			tokens, done, err := s.Submit(context.Background(),
				Request{Prompt: "x", MaxTokens: 4, Temperature: 1})
			if err != nil {
				t.Fatalf("submit %d: %v", i, err)
			}
			for range tokens {
			}
			if res := <-done; res.Err != nil {
				t.Fatalf("submit %d: %v", i, res.Err)
			}
		}
		check(t, s, "eight completed requests")
		if len(s.freeSlots) != slots {
			t.Fatalf("%d slots came back, want all %d: %v", len(s.freeSlots), slots, s.freeSlots)
		}
	})
}
