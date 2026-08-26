package sched

import (
	"sort"
	"sync"
	"time"
)

// latencies keeps the most recent completions so the server can report its own
// tail.
//
// An earlier version of this package said, in a comment two files over, that "a
// server that computes its own averages is a server that hides its tail" -- and
// then exposed a mean batch size and nothing else. The percentiles existed only
// in the load harness, which means they existed only when somebody was running
// a benchmark, which is exactly when a server's latency is least interesting.
//
// A window rather than a lifetime, because a p99 over every request since boot
// is a number that gets harder to move the longer the process runs: an hour of
// good service cannot pull it back, and an incident an hour ago never leaves it.
// What an operator wants is "how is it behaving now".
type latencies struct {
	mu    sync.Mutex
	ttft  []time.Duration
	total []time.Duration
	at    int
	full  bool
}

// latencyWindow is how many completions are kept. Sixteen kilobytes of
// durations, sorted on demand rather than maintained in order, because a sort
// of a thousand elements when somebody asks is cheaper than a heap kept correct
// on every request.
const latencyWindow = 1024

func newLatencies() *latencies {
	return &latencies{
		ttft:  make([]time.Duration, latencyWindow),
		total: make([]time.Duration, latencyWindow),
	}
}

// record takes one finished request. Only requests that produced a token are
// worth recording: a rejection has no time to first token, and mixing zeroes
// into the sample would drag every percentile down and make an overloaded
// server look fast.
func (l *latencies) record(res Result) {
	if res.Err != nil || res.Generated == 0 {
		return
	}

	l.mu.Lock()
	l.ttft[l.at] = res.FirstTok
	l.total[l.at] = res.Total
	l.at = (l.at + 1) % latencyWindow
	if l.at == 0 {
		l.full = true
	}
	l.mu.Unlock()
}

// snapshot returns the window's percentiles in milliseconds, and how many
// samples they were taken over -- because a p99 of four requests is not a p99,
// and a reader who cannot see the count cannot tell.
func (l *latencies) snapshot() LatencySnapshot {
	l.mu.Lock()
	n := l.at
	if l.full {
		n = latencyWindow
	}
	ttft := append([]time.Duration(nil), l.ttft[:n]...)
	total := append([]time.Duration(nil), l.total[:n]...)
	l.mu.Unlock()

	if n == 0 {
		return LatencySnapshot{}
	}
	sort.Slice(ttft, func(i, j int) bool { return ttft[i] < ttft[j] })
	sort.Slice(total, func(i, j int) bool { return total[i] < total[j] })

	return LatencySnapshot{
		Samples:     n,
		TTFTP50MS:   millis(ttft, 50),
		TTFTP95MS:   millis(ttft, 95),
		TTFTP99MS:   millis(ttft, 99),
		TotalP50MS:  millis(total, 50),
		TotalP95MS:  millis(total, 95),
		TotalP99MS:  millis(total, 99),
		WindowLimit: latencyWindow,
	}
}

// LatencySnapshot is what the server can say about its own tail, over the last
// Samples completions.
type LatencySnapshot struct {
	Samples     int     `json:"samples"`
	WindowLimit int     `json:"window"`
	TTFTP50MS   float64 `json:"ttft_p50_ms"`
	TTFTP95MS   float64 `json:"ttft_p95_ms"`
	TTFTP99MS   float64 `json:"ttft_p99_ms"`
	TotalP50MS  float64 `json:"total_p50_ms"`
	TotalP95MS  float64 `json:"total_p95_ms"`
	TotalP99MS  float64 `json:"total_p99_ms"`
}

// millis is the nearest-rank percentile of a sorted slice. No interpolation:
// interpolating between two measurements invents a third that nobody observed,
// which is the same rule the load harness follows so the two agree.
func millis(sorted []time.Duration, p int) float64 {
	if len(sorted) == 0 {
		return 0
	}
	rank := (p*len(sorted) + 99) / 100
	if rank < 1 {
		rank = 1
	}
	return float64(sorted[rank-1].Microseconds()) / 1000
}
