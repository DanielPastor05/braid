package api

import (
	"fmt"
	"net/http"
	"strings"

	"github.com/DanielPastor05/braid/internal/backend"
)

// The Prometheus text format, written by hand.
//
// The client library would be one dependency, and go.mod has none -- which is a
// decision this project has kept through an HTTP server, streaming, percentiles
// and a process pool, and is not worth spending on a format that is four lines
// of specification: a HELP line, a TYPE line, and `name{labels} value` per
// sample. Nothing here needs a registry, because there is exactly one scheduler
// and it already keeps the counters.
//
// What is deliberately *not* exported: the percentiles from /stats. Prometheus
// cannot aggregate a pre-computed quantile -- averaging two p99s gives a number
// that is not a percentile of anything -- so exporting them would produce
// dashboards that are wrong in a way nobody notices. The latency histogram below
// is the aggregatable form, and /stats keeps the exact percentiles for a human
// reading one server.
func (s *Server) metrics(w http.ResponseWriter, _ *http.Request) {
	var b strings.Builder
	snap := s.sched.Stats()

	counter := func(name, help string, value int64) {
		fmt.Fprintf(&b, "# HELP %s %s\n# TYPE %s counter\n%s %d\n", name, help, name, name, value)
	}
	gauge := func(name, help string, value float64) {
		fmt.Fprintf(&b, "# HELP %s %s\n# TYPE %s gauge\n%s %g\n", name, help, name, name, value)
	}

	counter("braid_requests_accepted_total", "Requests admitted to the queue.", snap.Queued)
	counter("braid_requests_completed_total", "Requests that generated every token asked for.", snap.Completed)
	counter("braid_requests_failed_total", "Requests that ended early, including cancellations.", snap.Failed)
	counter("braid_requests_rejected_total", "Requests refused because the queue was full.", snap.Rejected)
	counter("braid_requests_expired_total", "Requests whose deadline passed before they started.", snap.Expired)
	counter("braid_steps_total", "Forward passes.", snap.Steps)
	counter("braid_step_errors_total", "Forward passes the backend refused.", snap.StepErrors)
	counter("braid_sequences_advanced_total", "Sequences advanced, summed over steps.", snap.Advanced)
	counter("braid_tokens_total", "Tokens generated.", snap.Tokens)

	// The one number that says whether any of this worked. At 1.0 the server is
	// doing what serving one request at a time does, with more code.
	gauge("braid_mean_batch", "Sequences advanced per forward pass.", snap.MeanBatch)
	gauge("braid_mean_width", "Positions a step ran over: the longest sequence in it.", snap.MeanWidth)
	// Not a performance metric. It is the fraction of steps a position-keyed
	// key/value cache could serve at all, and it is why the cache this server
	// does have is indexed by slot instead.
	gauge("braid_aligned_step_share", "Steps whose sequences were all the same length.", snap.AlignedShare)

	// The second of these should never move. There are MaxBatch slots and at
	// most MaxBatch sequences active, so one is always free -- and a sequence
	// that somehow went without would still be served correctly, by
	// recomputing, which is exactly why nothing else would notice. Alert on it
	// being non-zero rather than on a rate.
	counter("braid_sequences_cached_total", "Sequences admitted with a cache slot.", snap.Cached)
	counter("braid_sequences_uncached_total",
		"Sequences admitted without a cache slot. Expected to stay at zero.", snap.Uncached)

	if l := snap.Latency; l.Samples > 0 {
		gauge("braid_latency_samples", "Completions in the latency window.", float64(l.Samples))
	}

	if t, ok := s.backend.(timed); ok {
		writeStepTimings(&b, t.Timings())
	}
	if p, ok := s.backend.(pooled); ok {
		stats := p.PoolStats()
		gauge("braid_workers_live", "Worker processes answering.", float64(stats.Live))
		gauge("braid_workers_configured", "Worker processes the pool was asked for.", float64(stats.Workers))
		counter("braid_worker_deaths_total", "Workers that stopped answering.", stats.Deaths)
		counter("braid_worker_failovers_total", "Steps retried on another worker.", stats.Failovers)
		counter("braid_worker_restarts_total", "Workers replaced after dying.", stats.Restarts)
	}

	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(b.String()))
}

// writeStepTimings exports where a step's time went, as gauges of milliseconds
// per step rather than as totals.
//
// A total would be the more Prometheus-shaped choice -- counters divide cleanly
// over a range -- but the worker reports these already averaged and the sum it
// averages is not exposed. Turning an average back into a total would be
// inventing precision, so they stay gauges and the HELP says so.
func writeStepTimings(b *strings.Builder, t backend.Timings) {
	if t.Steps == 0 {
		return
	}
	gauge := func(name, help string, value float64) {
		fmt.Fprintf(b, "# HELP %s %s\n# TYPE %s gauge\n%s %g\n", name, help, name, name, value)
	}
	gauge("braid_step_wall_ms", "Wall time of a step, averaged since start.", t.WallMS)
	gauge("braid_step_forward_ms", "The model's own time, averaged since start.", t.ForwardMS)
	gauge("braid_step_copy_ms", "Pulling the logits off the device, averaged.", t.CopyMS)
	gauge("braid_step_sample_ms", "Softmax and inverse CDF, averaged.", t.SampleMS)
	// Wall minus what the worker reports of itself. It can be negative when a
	// step is short enough that two clocks disagree, and it is exported as
	// measured rather than clamped, because a negative here means the
	// measurement is wrong and that is worth seeing on a graph.
	gauge("braid_step_pipe_ms", "The process boundary: wall time minus the worker's own accounting.", t.PipeMS)
	gauge("braid_step_kernels", "CUDA kernels a forward launched, averaged.", t.KernelsPerStep)
	gauge("braid_step_to_device", "Host-to-device copies per step.", t.ToDevicePerStep)
	gauge("braid_step_to_host", "Device-to-host copies per step.", t.ToHostPerStep)
}
