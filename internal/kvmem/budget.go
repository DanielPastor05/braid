// Package kvmem sizes the key/value cache from a memory budget.
//
// The scheduler used to admit requests by counting them, which is a fiction the
// queue can afford only while the cache does not exist: without one a sequence
// occupies a slot and nothing else. With a cache it occupies memory, and memory
// is the thing that runs out. At the model's 1024-id context, sixty-four
// sequences of keys and values across six blocks of six heads is 1.2 GB on a
// card with 8 -- so how many sequences may hold a cache at once is arithmetic
// about geometry rather than a constant somebody tuned.
//
// That arithmetic is all this package does, and it is what `-kv-budget-mb`
// asks. Sequences past the answer are served by recomputing.
//
// This package also held a block allocator, three eviction policies and a
// two-mode admission test -- the design for a paged cache, tested and called by
// nothing, because the worker's cache is slots x full context and nothing is
// ever allocated per block. They live on the `design/paged-attention` branch
// now. Using them means paged attention, which is a kernel this project does
// not have; keeping them here meant a package whose larger half no caller could
// reach.
package kvmem

import "fmt"

// Budget turns a card's spare memory into a number of blocks, and says what the
// numbers behind it were.
//
// The point is that the answer is arithmetic rather than a flag somebody tuned:
// a key/value cache is two tensors of (slots, heads, positions, head_dim) per
// block of the model, in float32, and that product is the whole of it. Writing
// it out means the size can be re-derived when the geometry changes instead of
// being a constant that used to be right.
type Budget struct {
	Heads     int
	HeadDim   int
	Layers    int
	Context   int // the model's full context, the most one sequence can hold
	BlockSize int // positions per block

	// Bytes is what the cache may occupy in total.
	Bytes int64
}

// BytesPerBlock is one block of one layer's keys and values, for every head.
func (b Budget) BytesPerBlock() int64 {
	const float32Bytes = 4
	perLayer := int64(b.Heads) * int64(b.HeadDim) * int64(b.BlockSize) * float32Bytes
	return perLayer * int64(b.Layers) * 2 // keys and values
}

// BytesPerSequence is what one sequence costs at the full context, which is the
// number that says whether a batch size is affordable at all.
func (b Budget) BytesPerSequence() int64 {
	return b.BytesPerBlock() * int64(b.blocksForContext())
}

// Blocks is how many the budget buys.
func (b Budget) Blocks() (int, error) {
	if b.Heads < 1 || b.HeadDim < 1 || b.Layers < 1 || b.Context < 1 || b.BlockSize < 1 {
		return 0, fmt.Errorf("kvmem: a budget needs a positive geometry, got %+v", b)
	}
	if b.Bytes < 1 {
		return 0, fmt.Errorf("kvmem: a budget of %d bytes buys nothing", b.Bytes)
	}

	blocks := b.Bytes / b.BytesPerBlock()
	if blocks < 1 {
		return 0, fmt.Errorf(
			"kvmem: %d bytes does not buy one %d-position block, which costs %d",
			b.Bytes, b.BlockSize, b.BytesPerBlock())
	}
	// A pool that cannot hold one sequence at full context can deadlock: a
	// sequence that grows past what is left has nothing to evict but itself.
	// Better to refuse at startup than to discover it under load.
	if blocks < int64(b.blocksForContext()) {
		return 0, fmt.Errorf(
			"kvmem: %d bytes buys %d blocks, and one sequence at the full %d-id context needs %d",
			b.Bytes, blocks, b.Context, b.blocksForContext())
	}
	return int(blocks), nil
}

// Sequences is how many could hold the full context at once, which is the honest
// answer to "what batch size does this memory support".
//
// At braid's geometry the answer deflates part of the reason this package
// exists, and it is better said than left for a reader to discover. One sequence
// at the full 1024-id context costs 18 MB; sixty-four cost 1.12 GB; four
// gigabytes of budget buys two hundred and twenty-seven of them. The server's
// measured peak is a batch of sixty, because the card runs out of *compute*
// first.
//
// So memory is not the binding constraint at this size, and the pressure this
// package is built to manage has to be created deliberately -- a smaller budget,
// or a longer context -- rather than found under ordinary load. That is a fact
// about a 10.7 M-parameter character model on an 8 GB card, not about the
// design: the same arithmetic at a real model's width and a 32k context is where
// it binds, and it binds hard.
func (b Budget) Sequences() int {
	blocks, err := b.Blocks()
	if err != nil {
		return 0
	}
	return blocks / b.blocksForContext()
}

func (b Budget) blocksForContext() int {
	return (b.Context + b.BlockSize - 1) / b.BlockSize
}
