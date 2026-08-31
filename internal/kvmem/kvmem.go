// Package kvmem hands out key/value cache in fixed-size blocks.
//
// The scheduler admits requests by counting them: QueueDepth is a number of
// requests, and a generation of a thousand tokens costs it exactly what a
// generation of ten costs. That is a fiction the queue can afford only while the
// cache does not exist, because without one a sequence occupies a slot and
// nothing else.
//
// With a cache it occupies memory, and memory is the thing that runs out. At the
// model's 1024-id context, sixty-four sequences of keys and values across six
// blocks of six heads is 1.2 GB on a card with 8 -- so what may be admitted is
// no longer "is there a slot" but "is there room for what this request said it
// would need".
//
// # Why blocks rather than one span per sequence
//
// A sequence does not know how long it will be. Reserving its maximum up front
// wastes whatever it does not use; reserving as it grows needs the reservation
// to be extendable, and a contiguous span cannot grow into whatever is next to
// it. Fixed-size blocks give up contiguity to get that: a sequence is a list of
// block indices, they need not be adjacent, and growing is taking one more from
// the free list.
//
// The cost is internal fragmentation, and it is the number this package reports
// rather than hides. A sequence of 17 positions at a block size of 16 holds two
// blocks and wastes fifteen slots. Bigger blocks mean less bookkeeping and more
// waste; the right size is a measurement, and Waste is what measures it.
package kvmem

import (
	"errors"
	"fmt"
)

// ErrFull is returned when a reservation cannot be met. It is a rejection and
// not a failure: the caller is expected to queue, evict or refuse, and which of
// those is the scheduler's business rather than the allocator's.
var ErrFull = errors.New("kvmem: no free blocks")

// Pool owns every block of the cache and the tables that say who holds which.
//
// It is not safe for concurrent use, which costs nothing here: the scheduler
// calls it from the one goroutine that owns the batch, and a mutex on this path
// would be a lock nothing ever contends.
type Pool struct {
	blockSize int
	blocks    int

	free   []int
	tables map[uint64][]int

	// lengths is what each sequence last asked for, which is not the same as
	// the capacity its blocks provide. Keeping it is what makes the waste
	// measurable rather than estimated.
	lengths map[uint64]int

	// held is the sum of those lengths. The difference between it and the
	// capacity the tables cover is the internal fragmentation.
	held int
}

// New builds a pool of `blocks` blocks, each holding `blockSize` positions.
func New(blocks, blockSize int) (*Pool, error) {
	if blocks < 1 {
		return nil, fmt.Errorf("kvmem: a pool needs at least one block, got %d", blocks)
	}
	if blockSize < 1 {
		return nil, fmt.Errorf("kvmem: a block needs at least one position, got %d", blockSize)
	}

	p := &Pool{
		blockSize: blockSize,
		blocks:    blocks,
		free:      make([]int, blocks),
		tables:    make(map[uint64][]int),
		lengths:   make(map[uint64]int),
	}
	// Handed out from the end, so the free list is a stack and a release is a
	// push. Which block a sequence gets does not matter -- that is the point of
	// them being uniform.
	for i := range p.free {
		p.free[i] = blocks - 1 - i
	}
	return p, nil
}

func (p *Pool) BlockSize() int { return p.blockSize }
func (p *Pool) Blocks() int    { return p.blocks }
func (p *Pool) FreeBlocks() int {
	return len(p.free)
}

// Fits reports whether a sequence of this many positions could be admitted now,
// without admitting it.
//
// Admission asks before it commits, because a request refused at the door costs
// nothing and one refused after its first token has cost a forward pass.
func (p *Pool) Fits(positions int) bool {
	return p.blocksFor(positions) <= len(p.free)
}

// Reserve grows a sequence's table to cover `positions`, taking blocks as
// needed. It is idempotent in the sense that reserving fewer positions than the
// sequence already holds is a no-op rather than a release: sequences only grow.
//
// Nothing is taken when the answer is ErrFull, so a failed reservation leaves
// the pool exactly as it was and the caller may retry after an eviction.
func (p *Pool) Reserve(id uint64, positions int) error {
	if positions < 0 {
		return fmt.Errorf("kvmem: negative positions (%d)", positions)
	}
	current := p.tables[id]
	want := p.blocksFor(positions)

	if need := want - len(current); need > 0 {
		if need > len(p.free) {
			return fmt.Errorf("%w: %d needed, %d free", ErrFull, need, len(p.free))
		}
		for range need {
			last := len(p.free) - 1
			current = append(current, p.free[last])
			p.free = p.free[:last]
		}
		p.tables[id] = current
	}

	// Sequences only grow. A reservation for fewer positions than the sequence
	// already has is a caller repeating itself, not a request to shrink -- and
	// shrinking would mean handing back a block whose keys are still being read.
	if positions > p.lengths[id] {
		p.held += positions - p.lengths[id]
		p.lengths[id] = positions
	}
	return nil
}

// Release returns every block a sequence holds. Releasing an unknown id is not
// an error: a sequence that failed before it reserved anything still gets
// released on the way out, and making that a special case at every call site is
// how a leak gets written.
func (p *Pool) Release(id uint64) {
	blocks, held := p.tables[id]
	if !held {
		return
	}
	p.free = append(p.free, blocks...)
	delete(p.tables, id)
	p.held -= p.lengths[id]
	delete(p.lengths, id)
}

// Table is the blocks a sequence holds, in order. The slice is the pool's: the
// caller may read it and must not keep it past the next Reserve.
func (p *Pool) Table(id uint64) []int { return p.tables[id] }

// Waste is the share of allocated capacity nobody is using: the internal
// fragmentation that fixed-size blocks buy growability with.
//
// Zero when every sequence happens to end on a block boundary, and at its worst
// (blockSize-1)/blockSize when they all end just past one. It is reported rather
// than assumed because the right block size is a trade between this and the
// bookkeeping, and a trade needs both numbers.
func (p *Pool) Waste() float64 {
	capacity := 0
	for _, blocks := range p.tables {
		capacity += len(blocks) * p.blockSize
	}
	if capacity == 0 {
		return 0
	}
	return float64(capacity-p.held) / float64(capacity)
}

// Utilisation is the share of the pool's blocks that are held by somebody.
func (p *Pool) Utilisation() float64 {
	return float64(p.blocks-len(p.free)) / float64(p.blocks)
}

func (p *Pool) blocksFor(positions int) int {
	if positions <= 0 {
		return 0
	}
	return (positions + p.blockSize - 1) / p.blockSize
}

// freeList exposes the free blocks for the invariant test, which has to see
// every block in the pool to prove none is in two places at once.
func (p *Pool) freeList() []int { return p.free }
