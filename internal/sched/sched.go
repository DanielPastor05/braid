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

	// ErrWouldMissDeadline is returned at submission when the queue is short
	// enough to accept the request and long enough that it would not be served
	// in time.
	//
	// It exists because of a measurement. Offered two hundred requests a second
	// against a capacity near one hundred and fifty, this server admitted every
	// one of them -- the queue is 256 deep and swallowed the whole overload --
	// and answered them at a p99 of 1.3 seconds against a 100 ms budget, without
	// a single rejection to say so. Queue depth is the wrong thing to admit on:
	// it says whether there is *room*, and the caller asked about *time*.
	ErrWouldMissDeadline = errors.New("sched: the queue is longer than the caller's deadline")

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

// Default was a starting point and is now a measurement, at least for MaxBatch.
//
// It was 32 because 32 is a round number. Swept against the real backend at 32,
// 64 and 128, throughput saturates at about 2 750 tokens/s from a batch of
// roughly sixty and does not rise after: 64 buys 8% over 32 and 128 buys nothing
// over 64. What the limit actually controls past that point is whether clients
// queue. At 64 concurrent clients, raising it from 32 to 64 left the throughput
// where it was and took the median time to first token from 382 ms to 47 ms,
// because a client that does not fit in the batch is a client waiting for a slot
// rather than a client being served slowly.
//
// So it is a latency knob above the saturation point, not a throughput one, and
// 64 is where it costs nothing at low concurrency and removes the cliff at high.
// Going further is not free: a batch of 128 makes a step 41 ms where 64 makes it
// 21, and everybody in that batch waits the whole step.
//
// QueueDepth and MaxTokensLimit are still round numbers, and this comment is
// still the place that will say so when they stop being.
func Default() Config {
	return Config{MaxBatch: 64, QueueDepth: 256, MaxTokensLimit: 1024}
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

	// Which key/value cache slot in the worker this sequence owns, or -1 for
	// none. Held for the sequence's whole life and returned on the way out, so
	// that a slot's contents stay the history of one sequence and the worker can
	// tell "this cache is yours and current" from "refill it".
	//
	// A slot is not authority over anything: the scheduler still sends the whole
	// window every step, so a worker without the cache -- a replacement after a
	// failover -- recomputes and is right.
	slot int32
}

// window writes the model's fixed-width view of the sequence into dst -- the
// last SeqLen ids, at the *front*, padded to width with zeros -- and returns how
// many of them are real.
//
// The padding is on the right because the causal mask only hides the future,
// not the padding. Left-padded, the position being sampled attends to every pad
// id before it; right-padded, position take-1 attends to 0..take-1 and the pad
// beyond it can never be reached. The difference is not academic: id 0 in this
// alphabet is a tab, so an eleven-character prompt used to arrive at the model
// as two hundred and forty-five tabs followed by the prompt, and the model
// answered the tabs.
func (s *sequence) window(dst []int32) int {
	for i := range dst {
		dst[i] = 0
	}
	take := len(s.history)
	if take > len(dst) {
		take = len(dst)
	}
	copy(dst, s.history[len(s.history)-take:])
	if take == 0 {
		// An empty prompt, or one made entirely of bytes the model was never
		// trained on. There is no such thing as a forward pass over nothing and
		// no such thing as sampling position -1, so the sequence starts from a
		// single pad id and generates what follows it, which is a real answer to
		// a real request.
		return 1
	}
	return take
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

	// Free key/value cache slots in the worker, one per row of MaxBatch. Only
	// the loop goroutine touches it, which is why it needs no lock -- the same
	// reason nothing else in the scheduler's hot path has one.
	freeSlots []int32
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

	// Handed out from the end, so the first sequence gets slot 0 and the
	// numbering a worker sees starts where a person would expect it to.
	free := make([]int32, cfg.MaxBatch)
	for i := range free {
		free[i] = int32(cfg.MaxBatch - 1 - i)
	}

	s := &Scheduler{
		cfg:       cfg,
		backend:   b,
		seqLen:    b.SeqLen(),
		freeSlots: free,
		incoming:  make(chan *sequence, cfg.QueueDepth),
		stop:      make(chan struct{}),
		stopped:   make(chan struct{}),
		latency:   newLatencies(),
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

	// Told no now rather than late later. Only for callers who said what late
	// means to them: without a MaxWait there is no deadline to miss and the
	// queue is the right answer.
	if req.MaxWait > 0 && s.predictedWait(req.MaxTokens) > req.MaxWait {
		s.stats.rejected.Add(1)
		return nil, nil, ErrWouldMissDeadline
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

// predictedWait estimates how long a request arriving now would sit before its
// first token.
//
// At capacity the batch is full, so a queued request waits for somebody to
// finish. One sequence finishes every `tokens / MaxBatch` steps on average, so
// the nth in the queue waits about n of those. The new request's own MaxTokens
// stands in for the queue's, which is wrong in detail and right in shape: a
// queue of long generations does drain more slowly.
//
// Deliberately not a control loop. It is one multiplication over counters the
// scheduler already keeps, and it is only ever compared against a deadline the
// caller volunteered -- so being off by a factor is a request served or refused
// slightly wrongly, not a server that oscillates.
func (s *Scheduler) predictedWait(tokens int) time.Duration {
	steps := s.stats.steps.Load()
	if steps == 0 {
		return 0 // nothing measured yet; the queue is the only answer available
	}
	perStep := time.Duration(s.stats.stepNanos.Load() / steps)
	ahead := len(s.incoming)
	if ahead == 0 {
		return 0
	}
	stepsPerCompletion := float64(tokens) / float64(s.cfg.MaxBatch)
	return time.Duration(float64(ahead) * stepsPerCompletion * float64(perStep))
}

// Close stops the loop and fails everything still in flight. It returns once
// the loop has exited.
func (s *Scheduler) Close() error {
	s.closeOne.Do(func() { close(s.stop) })
	<-s.stopped
	return s.backend.Close()
}
