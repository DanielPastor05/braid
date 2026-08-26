// Package sched merges independent requests into shared batches.
//
// The engine behind this server generates one token per forward pass over a
// fixed 256-id window. Serving one request at a time means one forward pass per
// token per client, and a GPU that spends most of its time waiting for the next
// HTTP request rather than computing. Serving them in a static batch means
// everybody waits for the slowest member to finish before anybody new can start.
//
// This is the third option. There is one loop, it holds a set of sequences, and
// on every pass it stacks their windows into a single (n, 256) tensor and
// advances all of them by exactly one token. A request that arrives mid-flight
// joins at the next step; a request that finishes leaves at the next step; the
// others do not notice either event. That last clause is the correctness
// property, and TestBatchingDoesNotChangeOutput holds the implementation to it.
package sched

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/DanielPastor05/braid/internal/backend"
)

var (
	// ErrQueueFull is returned at admission when the queue is at its depth.
	// It is a rejection, not a delay: a server that queues without bound
	// converts a throughput problem into a latency problem and reports neither.
	ErrQueueFull = errors.New("sched: queue is full")

	// ErrDeadlineExceeded is returned when a request could not start before the
	// deadline its caller set. Rejecting at admission costs nothing; noticing
	// after the GPU has already worked on it wastes the work.
	ErrDeadlineExceeded = errors.New("sched: deadline passed before the request could start")

	// ErrClosed is returned once the scheduler has been shut down.
	ErrClosed = errors.New("sched: closed")
)

// Request is one generation.
type Request struct {
	Prompt      string
	MaxTokens   int
	Temperature float32
	Seed        uint64

	// MaxWait bounds the time this request will sit in the queue before it is
	// rejected. Zero means it waits as long as it takes.
	MaxWait time.Duration
}

// Token is one generated token, delivered as it is produced.
type Token struct {
	ID   int32
	Text string
}

// Result is delivered once, after the token channel closes.
type Result struct {
	// Err is non-nil if generation stopped early: a backend failure, or the
	// caller's context being cancelled.
	Err error

	Queued    time.Duration // admission wait
	FirstTok  time.Duration // from submission to the first token
	Total     time.Duration // from submission to the last token
	Generated int
}

// Config tunes the loop. The zero value is not usable; see Default.
type Config struct {
	// MaxBatch is the most sequences that may share one forward pass. Above the
	// point where the GPU is saturated this buys nothing and costs latency for
	// everyone in the batch, which is why it is a number and not "as many as
	// arrive".
	MaxBatch int

	// QueueDepth is how many admitted-but-not-started requests may wait. This
	// is the backpressure valve: past it, callers are told no.
	QueueDepth int

	// MaxTokensLimit is the longest generation a caller may ask for. It bounds
	// two things at once: how long one sequence can hold a slot in the batch,
	// and how much memory its stream can buffer, since the two are the same
	// number. See the note on the output channel in Submit.
	MaxTokensLimit int
}

// Default is a starting point, not a measurement. The right MaxBatch is the one
// past which throughput stops rising and only the tail moves, and that number
// belongs to a backend -- so it cannot be known until there is a real one to
// sweep. Until then these are round numbers, and the README says so.
func Default() Config {
	return Config{MaxBatch: 32, QueueDepth: 256, MaxTokensLimit: 1024}
}

// A sequence is one in-flight request inside the loop.
type sequence struct {
	req     Request
	ctx     context.Context
	history []int32 // prompt followed by everything generated
	out     chan Token
	done    chan Result

	submitted time.Time
	admitted  time.Time
	first     time.Time
	generated int
}

// window returns the model's fixed-width view of the sequence: the last SeqLen
// ids, left-padded with zeros when the history is shorter. It writes into dst
// so the loop can reuse one backing array across steps.
func (s *sequence) window(dst []int32) {
	for i := range dst {
		dst[i] = 0
	}
	take := len(s.history)
	if take > len(dst) {
		take = len(dst)
	}
	copy(dst[len(dst)-take:], s.history[len(s.history)-take:])
}

// Scheduler owns a backend and the single goroutine that drives it.
type Scheduler struct {
	cfg     Config
	backend backend.Backend
	seqLen  int

	incoming chan *sequence
	stop     chan struct{}
	stopped  chan struct{}
	closeOne sync.Once

	stats   Stats
	latency *latencies
}

// New starts the loop. Close stops it.
func New(b backend.Backend, cfg Config) (*Scheduler, error) {
	if cfg.MaxBatch < 1 {
		return nil, fmt.Errorf("sched: MaxBatch must be at least 1, got %d", cfg.MaxBatch)
	}
	if cfg.QueueDepth < 1 {
		return nil, fmt.Errorf("sched: QueueDepth must be at least 1, got %d", cfg.QueueDepth)
	}
	if cfg.MaxTokensLimit < 1 {
		return nil, fmt.Errorf("sched: MaxTokensLimit must be at least 1, got %d", cfg.MaxTokensLimit)
	}

	s := &Scheduler{
		cfg:      cfg,
		backend:  b,
		seqLen:   b.SeqLen(),
		incoming: make(chan *sequence, cfg.QueueDepth),
		stop:     make(chan struct{}),
		stopped:  make(chan struct{}),
		latency:  newLatencies(),
	}
	go s.loop()
	return s, nil
}

// Submit hands a request to the loop and returns the stream it will be written
// to. The token channel is closed when generation ends, and exactly one Result
// follows on the second channel.
//
// It returns an error only for rejections that happen before any work is done:
// a full queue, a deadline already passed, a closed scheduler.
func (s *Scheduler) Submit(ctx context.Context, req Request) (<-chan Token, <-chan Result, error) {
	if req.MaxTokens < 1 || req.MaxTokens > s.cfg.MaxTokensLimit {
		return nil, nil, fmt.Errorf("sched: MaxTokens must be between 1 and %d, got %d",
			s.cfg.MaxTokensLimit, req.MaxTokens)
	}
	if req.Temperature <= 0 {
		return nil, nil, fmt.Errorf("sched: Temperature must be above zero, got %v", req.Temperature)
	}

	// Only the last SeqLen ids of a prompt can ever be read -- window() takes
	// the tail and the model has no other context -- so keeping the rest is a
	// caller-controlled allocation with no purpose. At a 1 MiB body limit and a
	// queue of 256 that was up to a gigabyte of ids nothing would ever look at.
	//
	// Copied rather than resliced: history[len-n:] keeps the whole backing array
	// alive, which would have fixed the arithmetic and none of the memory.
	history := s.backend.Encode(req.Prompt)
	if len(history) > s.seqLen {
		history = append([]int32(nil), history[len(history)-s.seqLen:]...)
	}

	seq := &sequence{
		req:     req,
		ctx:     ctx,
		history: history,
		// The buffer holds every token this request could possibly produce, so
		// the loop's send can never block and never has to decide what to do
		// about a caller that has stopped reading. A stalled caller pays for
		// itself in memory -- MaxTokens tokens, bounded by MaxTokensLimit --
		// and costs the sequences sharing its batch nothing at all. That trade
		// is why the loop has no slow-consumer policy: it does not need one.
		out:       make(chan Token, req.MaxTokens),
		done:      make(chan Result, 1),
		submitted: time.Now(),
	}

	select {
	case <-s.stop:
		return nil, nil, ErrClosed
	default:
	}

	select {
	case s.incoming <- seq:
		s.stats.queued.Add(1)
		return seq.out, seq.done, nil
	default:
		s.stats.rejected.Add(1)
		return nil, nil, ErrQueueFull
	}
}

// Close stops the loop and fails everything still in flight. It returns once
// the loop has exited.
func (s *Scheduler) Close() error {
	s.closeOne.Do(func() { close(s.stop) })
	<-s.stopped
	return s.backend.Close()
}
