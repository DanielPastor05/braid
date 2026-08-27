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
		Steps:      s.stats.steps.Load(),
		Advanced:   s.stats.advanced.Load(),
		Tokens:     s.stats.tokens.Load(),
		Queued:     s.stats.queued.Load(),
		Rejected:   s.stats.rejected.Load(),
		Expired:    s.stats.expired.Load(),
		Completed:  s.stats.completed.Load(),
		Failed:     s.stats.failed.Load(),
		StepErrors: s.stats.stepErrors.Load(),
	}
	if snap.Steps > 0 {
		snap.MeanBatch = float64(snap.Advanced) / float64(snap.Steps)
		snap.AlignedShare = float64(s.stats.alignedSteps.Load()) / float64(snap.Steps)
		snap.MeanWidth = float64(s.stats.widthSum.Load()) / float64(snap.Steps)
	}
	snap.Latency = s.latency.snapshot()
	return snap
}
