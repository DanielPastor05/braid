package sched

import "sync/atomic"

// Stats counts what the loop did. Every counter is a total since start; rates
// are the caller's business, because a rate needs two readings and this has one.
//
// Percentiles are not the caller's business, and an earlier version of this
// comment said they were. They lived only in the load harness, which meant the
// server could describe its own tail only while somebody was benchmarking it --
// precisely when the tail matters least. See latency.go.
type Stats struct {
	steps      atomic.Int64
	advanced   atomic.Int64
	tokens     atomic.Int64
	queued     atomic.Int64
	rejected   atomic.Int64
	expired    atomic.Int64
	completed  atomic.Int64
	failed     atomic.Int64
	stepErrors atomic.Int64

	// alignedSteps counts the steps whose sequences were all the same length,
	// and widthSum the width each step actually ran at.
	//
	// Neither is a performance counter. They are the two numbers that say what a
	// key/value cache could do here, and a cache is the reason they exist. A
	// cache keyed by position needs every row of the batch at the same position,
	// because one write offset is shared by the whole batch -- so alignedSteps
	// over steps is the fraction of steps such a cache could serve at all. The
	// width is what the batch costs instead: the longest row, since every row is
	// computed to the same width.
	alignedSteps atomic.Int64
	widthSum     atomic.Int64

	// cached counts sequences admitted with a key/value cache slot, uncached
	// those admitted without one, and which of the two is interesting depends on
	// CacheSlots.
	//
	// At a slot per row of the batch, uncached cannot move: a sequence takes one
	// in begin(), begin() is only reached while fewer than MaxBatch are active,
	// and every active sequence holds exactly one until it finishes. Live slots
	// equal the active count, so the free list is never empty. There, zero is an
	// assertion -- if it moves, the free list is leaking.
	//
	// Below that, it moves on purpose. Cache memory is a budget, the sequences
	// that miss out recompute, and the ratio is what the budget bought or gave
	// up.
	cached   atomic.Int64
	uncached atomic.Int64

	// freeSlots mirrors len(Scheduler.freeSlots), because the list itself is
	// owned by the loop goroutine and reading it from Stats() would be a race.
	//
	// It exists because the two counters above watch the wrong direction. They
	// notice a free list that *shrinks* -- sequences start recomputing, uncached
	// moves. A list that *grows* is invisible to them: everyone still gets a
	// slot, uncached stays at zero, and the same row is handed to several live
	// sequences at once. That is not a slowdown, it is the two of them reading
	// each other's keys, and it is exactly the bug that shipped here.
	//
	// So this is the one to alert on, and the condition is an equality rather
	// than a threshold: free plus live must equal the number of slots that
	// exist, always. Past that number there is nothing to tune, only a bug.
	freeSlots atomic.Int64

	// stepNanos is the total time spent in Step, for estimating how long a
	// queued request would wait. It is a sum rather than an average because two
	// atomics cannot be read together and a ratio of two counters can be.
	stepNanos atomic.Int64
}

// Snapshot is Stats at one instant, safe to serialise.
type Snapshot struct {
	Steps      int64 `json:"steps"`
	Advanced   int64 `json:"sequences_advanced"`
	Tokens     int64 `json:"tokens"`
	Queued     int64 `json:"accepted"`
	Rejected   int64 `json:"rejected_queue_full"`
	Expired    int64 `json:"rejected_deadline"`
	Completed  int64 `json:"completed"`
	Failed     int64 `json:"failed"`
	StepErrors int64 `json:"step_errors"`

	// AlignedShare is the fraction of steps whose sequences were all the same
	// length. See the note on alignedSteps: it is what a position-keyed cache
	// could serve, and under continuous batching it is not 1.
	AlignedShare float64 `json:"aligned_step_share"`
	// MeanWidth is the positions a step actually ran over, which is the longest
	// sequence in it and not the model's full context.
	MeanWidth float64 `json:"mean_width"`

	// Cached and Uncached are sequences admitted with and without a key/value
	// cache slot. See the note on the counters: whether the second one is an
	// assertion or a measurement depends on whether CacheSlots is below
	// MaxBatch.
	Cached   int64 `json:"cached_sequences"`
	Uncached int64 `json:"uncached_sequences"`

	// FreeSlots is how many cache slots are not held right now, and CacheSlots
	// how many exist. The invariant is that free never exceeds the total, and
	// the two are reported together so that it can be checked without knowing
	// how the server was configured. See the note on the counter.
	FreeSlots  int64 `json:"free_cache_slots"`
	CacheSlots int64 `json:"cache_slots"`

	// MeanStepMillis is how long a forward pass has been taking. It exists to
	// predict queueing, and it is reported because a prediction whose inputs
	// are invisible cannot be argued with.
	MeanStepMillis float64 `json:"mean_step_ms"`

	// Latency over a recent window, not since boot. A p99 measured over every
	// request a long-lived process ever served is a number an hour of good
	// service cannot move.
	Latency LatencySnapshot `json:"latency"`

	// MeanBatch is sequences advanced per step. It is the one number that says
	// whether any of this worked: at 1.0 the server is doing exactly what
	// serving one request at a time does, with more code.
	MeanBatch float64 `json:"mean_batch"`
}

// Stats returns a consistent-enough snapshot: the counters are read one after
// another rather than under a lock, so a snapshot taken mid-step can show a
// step that has not finished being accounted for. That is worth a nanosecond of
// skew rather than a mutex on the hot path.
func (s *Scheduler) Stats() Snapshot {
	snap := Snapshot{
		Cached:     s.stats.cached.Load(),
		Uncached:   s.stats.uncached.Load(),
		Steps:      s.stats.steps.Load(),
		Advanced:   s.stats.advanced.Load(),
		Tokens:     s.stats.tokens.Load(),
		Queued:     s.stats.queued.Load(),
		Rejected:   s.stats.rejected.Load(),
		Expired:    s.stats.expired.Load(),
		Completed:  s.stats.completed.Load(),
		Failed:     s.stats.failed.Load(),
		StepErrors: s.stats.stepErrors.Load(),
		FreeSlots:  s.stats.freeSlots.Load(),
		CacheSlots: int64(s.cacheSlots),
	}
	if snap.Steps > 0 {
		snap.MeanStepMillis = float64(s.stats.stepNanos.Load()) / float64(snap.Steps) / 1e6
		snap.MeanBatch = float64(snap.Advanced) / float64(snap.Steps)
		snap.AlignedShare = float64(s.stats.alignedSteps.Load()) / float64(snap.Steps)
		snap.MeanWidth = float64(s.stats.widthSum.Load()) / float64(snap.Steps)
	}
	snap.Latency = s.latency.snapshot()
	return snap
}
