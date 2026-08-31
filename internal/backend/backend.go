// Package backend is the seam between the scheduler and whatever actually runs
// the model.
//
// The whole contract is one blocking call that advances a batch of sequences by
// exactly one token. Everything the scheduler does interesting -- merging
// independent requests into a shared batch, admitting and evicting sequences
// between steps, backpressure, cancellation -- sits on top of this and needs no
// GPU to be exercised. That is deliberate: the batching logic is the part worth
// getting right, and it is testable to the last branch against Mock.
package backend

import "context"

// Backend advances sequences by one token each.
//
// Implementations are not required to be safe for concurrent use. The scheduler
// owns a backend and calls it from a single goroutine.
type Backend interface {
	// Step takes one window of token ids per sequence and returns the next
	// token id for each, in the same order. Every window is exactly SeqLen
	// ids long, padded with id 0 on the right when the sequence is shorter.
	//
	// lengths says how many ids of each window are real. The rest is padding on
	// the right, which the causal mask makes unreachable from the position being
	// sampled -- and which is why the caller has to say where that position is
	// rather than the backend assuming the last one.
	//
	// The context bounds how long the caller will wait. It matters because the
	// interesting failure is not a backend that dies -- a dead process closes
	// its pipe and the read fails at once -- but one that is alive and silent:
	// a GPU hang, a driver reset that leaves the process up. Without a deadline
	// that stops the scheduler for good, with every request behind it, while
	// the health check goes on saying the server is fine.
	//
	// A batch of n sequences must produce the same n results as n batches of
	// one -- the sequences share a tensor, not a computation. The scheduler's
	// tests hold implementations to that.
	Step(ctx context.Context, windows [][]int32, lengths []int32, slots []int32,
		temperatures []float32, seeds []uint64) ([]int32, error)

	// SeqLen is the fixed context width the model was built with. The engine's
	// character model uses 256.
	SeqLen() int

	// VocabSize is the number of distinct token ids.
	VocabSize() int

	// Encode turns text into token ids, dropping anything outside the vocabulary.
	Encode(text string) []int32

	// Decode turns token ids back into text.
	Decode(ids []int32) string

	// Close releases whatever the implementation is holding.
	Close() error
}
