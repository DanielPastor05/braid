package sched

import (
	"context"
	"time"
)

// loop is the whole scheduler. It runs in one goroutine, owns every sequence in
// flight, and is the only thing that touches the backend.
func (s *Scheduler) loop() {
	defer close(s.stopped)

	active := make([]*sequence, 0, s.cfg.MaxBatch)

	// Cancelled when the scheduler is closed, so a shutdown does not wait on a
	// backend that may never answer. The per-step deadline is the backend's
	// own: it knows what its steps cost and this loop does not.
	stepCtx, cancelSteps := context.WithCancel(context.Background())
	defer cancelSteps()
	go func() {
		<-s.stop
		cancelSteps()
	}()

	// One backing array for the batch, reused across steps. The rows handed to
	// the backend are windows into it, so a step of n sequences allocates
	// nothing beyond the slice headers.
	flat := make([]int32, s.cfg.MaxBatch*s.seqLen)
	windows := make([][]int32, 0, s.cfg.MaxBatch)
	lengths := make([]int32, 0, s.cfg.MaxBatch)
	slots := make([]int32, 0, s.cfg.MaxBatch)
	temps := make([]float32, 0, s.cfg.MaxBatch)
	seeds := make([]uint64, 0, s.cfg.MaxBatch)

	for {
		active = s.admit(active)

		if len(active) == 0 {
			// Nothing to do: wait for work rather than spinning on an empty
			// batch. This is the only place the loop blocks.
			select {
			case seq := <-s.incoming:
				if kept := s.begin(seq); kept != nil {
					active = append(active, kept)
				}
			case <-s.stop:
				return
			}
			continue
		}

		select {
		case <-s.stop:
			s.failAll(active, ErrClosed)
			return
		default:
		}

		windows = windows[:0]
		lengths = lengths[:0]
		slots = slots[:0]
		temps = temps[:0]
		seeds = seeds[:0]
		for i, seq := range active {
			w := flat[i*s.seqLen : (i+1)*s.seqLen]
			lengths = append(lengths, int32(seq.window(w)))
			windows = append(windows, w)
			slots = append(slots, seq.slot)
			temps = append(temps, seq.req.Temperature)
			// The seed advances with the sequence, so a sequence is
			// reproducible on its own and two sequences sharing a seed do not
			// walk in lockstep once they diverge.
			seeds = append(seeds, seq.req.Seed+uint64(seq.generated))
		}

		startedStep := time.Now()
		ids, err := s.backend.Step(stepCtx, windows, lengths, slots, temps, seeds)
		s.stats.stepNanos.Add(int64(time.Since(startedStep)))
		if err != nil {
			s.stats.stepErrors.Add(1)
			s.failAll(active, err)
			active = active[:0]
			continue
		}

		s.stats.steps.Add(1)
		s.stats.advanced.Add(int64(len(active)))

		// What this step cost in positions, and whether a position-keyed cache
		// could have served it. Counted here rather than in the worker because
		// the scheduler already knows both, and asking the worker would mean a
		// protocol field for a measurement.
		width, aligned := int32(0), true
		for _, length := range lengths {
			if length > width {
				width = length
			}
			if length != lengths[0] {
				aligned = false
			}
		}
		s.stats.widthSum.Add(int64(width))
		if aligned {
			s.stats.alignedSteps.Add(1)
		}

		now := time.Now()
		kept := active[:0]
		for i, seq := range active {
			if done := s.deliver(seq, ids[i], now); !done {
				kept = append(kept, seq)
			}
		}
		active = kept
	}
}

// deliver appends one token to a sequence and reports whether it is finished.
func (s *Scheduler) deliver(seq *sequence, id int32, now time.Time) bool {
	if err := seq.ctx.Err(); err != nil {
		s.finish(seq, err)
		return true
	}

	seq.history = append(seq.history, id)
	seq.generated++
	if seq.generated == 1 {
		seq.first = now
	}

	// This send cannot block: the channel was made with room for MaxTokens
	// tokens and this is the generated-th of at most MaxTokens. The loop
	// therefore never waits on a caller, which is the whole point of sizing the
	// buffer that way rather than policing whoever reads slowly.
	seq.out <- Token{ID: id, Text: s.backend.Decode([]int32{id})}

	s.stats.tokens.Add(1)
	if seq.generated >= seq.req.MaxTokens {
		s.finish(seq, nil)
		return true
	}
	return false
}

// admit moves waiting requests into the batch, without blocking. The loop must
// never stall on an empty queue while it has sequences to advance.
func (s *Scheduler) admit(active []*sequence) []*sequence {
	for len(active) < s.cfg.MaxBatch {
		select {
		case seq := <-s.incoming:
			if kept := s.begin(seq); kept != nil {
				active = append(active, kept)
			}
		default:
			return active
		}
	}
	return active
}

// begin checks a request that has reached the front of the queue and is about
// to cost GPU time. It returns nil if the request should not start.
func (s *Scheduler) begin(seq *sequence) *sequence {
	now := time.Now()

	if err := seq.ctx.Err(); err != nil {
		s.finish(seq, err)
		return nil
	}
	if seq.req.MaxWait > 0 && now.Sub(seq.submitted) > seq.req.MaxWait {
		s.stats.expired.Add(1)
		s.finish(seq, ErrDeadlineExceeded)
		return nil
	}

	seq.admitted = now
	seq.slot = s.takeSlot()
	if seq.slot < 0 {
		s.stats.uncached.Add(1)
	} else {
		s.stats.cached.Add(1)
	}
	return seq
}

func (s *Scheduler) failAll(seqs []*sequence, err error) {
	for _, seq := range seqs {
		s.finish(seq, err)
	}
}

// finish closes a sequence's stream and reports it exactly once.
func (s *Scheduler) finish(seq *sequence, err error) {
	s.releaseSlot(seq)
	close(seq.out)

	end := time.Now()
	res := Result{Err: err, Generated: seq.generated, Total: end.Sub(seq.submitted)}
	if !seq.admitted.IsZero() {
		res.Queued = seq.admitted.Sub(seq.submitted)
	}
	if !seq.first.IsZero() {
		res.FirstTok = seq.first.Sub(seq.submitted)
	}

	if err != nil {
		s.stats.failed.Add(1)
	} else {
		s.stats.completed.Add(1)
	}
	s.latency.record(res)
	seq.done <- res
	close(seq.done)
}

// Cache slots.
//
// A slot is a row of the worker's key/value cache, and there are MaxBatch of
// them because that is the most sequences that can share a step. A sequence
// takes one when it is admitted and gives it back when it ends, so a slot's
// contents belong to one sequence for that sequence's whole life -- which is
// what lets the worker tell "this cache is current" from "refill it" by
// comparing what it holds against the row's length.
//
// Handing them out in reverse order is not a preference, it is what makes the
// free list a stack: pop from the end, push to the end. Which slot a sequence
// gets does not matter, only that no two hold the same one at once.
//
// -1 means no slot, and the worker recomputes that row. It happens when more
// sequences are somehow live than there are slots, and it is a slower answer
// rather than a wrong one.
func (s *Scheduler) takeSlot() int32 {
	if len(s.freeSlots) == 0 {
		return -1
	}
	slot := s.freeSlots[len(s.freeSlots)-1]
	s.freeSlots = s.freeSlots[:len(s.freeSlots)-1]
	return slot
}

func (s *Scheduler) releaseSlot(seq *sequence) {
	if seq.slot < 0 {
		return
	}
	s.freeSlots = append(s.freeSlots, seq.slot)
	// Cleared so that a double release -- which would hand the same slot to two
	// sequences and quietly give one of them the other's history -- is a bug
	// that cannot happen rather than one that is unlikely.
	seq.slot = -1
}
