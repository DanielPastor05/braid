package sched

import "sync/atomic"

// Stats counts what the loop did. Every field is a total since start; rates and
// percentiles are the caller's business, because a server that computes its own
// averages is a server that hides its tail.
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
	}
	return snap
}
