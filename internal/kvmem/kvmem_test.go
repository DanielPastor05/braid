package kvmem

import (
	"errors"
	"math"
	"math/rand/v2"
	"testing"
)

func newPool(t *testing.T, blocks, blockSize int) *Pool {
	t.Helper()
	p, err := New(blocks, blockSize)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestAReservationTakesTheBlocksItNeedsAndNoMore(t *testing.T) {
	p := newPool(t, 10, 16)

	cases := []struct {
		positions int
		blocks    int
	}{
		{1, 1},  // one position still costs a whole block
		{16, 1}, // exactly a block
		{17, 2}, // one past, and the second block is almost all waste
		{32, 2}, // exactly two
		{0, 0},  // nothing costs nothing
	}
	for _, c := range cases {
		p := newPool(t, 10, 16)
		if err := p.Reserve(1, c.positions); err != nil {
			t.Fatalf("%d positions: %v", c.positions, err)
		}
		if got := len(p.Table(1)); got != c.blocks {
			t.Errorf("%d positions took %d blocks, want %d", c.positions, got, c.blocks)
		}
		if got := p.FreeBlocks(); got != 10-c.blocks {
			t.Errorf("%d positions left %d free, want %d", c.positions, got, 10-c.blocks)
		}
	}
	_ = p
}

// TestGrowingKeepsTheBlocksItHas is the property a generation depends on: a
// sequence reserves again on every token, and it must not re-acquire what it
// already holds.
func TestGrowingKeepsTheBlocksItHas(t *testing.T) {
	p := newPool(t, 10, 16)

	if err := p.Reserve(1, 10); err != nil {
		t.Fatal(err)
	}
	first := append([]int(nil), p.Table(1)...)

	// Nine more tokens, all inside the first block.
	for n := 11; n <= 16; n++ {
		if err := p.Reserve(1, n); err != nil {
			t.Fatalf("growing to %d: %v", n, err)
		}
	}
	if got := len(p.Table(1)); got != 1 {
		t.Errorf("staying inside one block took %d blocks", got)
	}

	// The seventeenth crosses the boundary.
	if err := p.Reserve(1, 17); err != nil {
		t.Fatal(err)
	}
	table := p.Table(1)
	if len(table) != 2 {
		t.Fatalf("crossing a boundary took %d blocks, want 2", len(table))
	}
	if table[0] != first[0] {
		t.Errorf("growing moved the first block from %d to %d; the keys in it are still being read",
			first[0], table[0])
	}
}

// TestAFailedReservationTakesNothing is the one that matters for eviction. The
// caller is expected to free something and try again, and that only works if the
// failed attempt left the pool exactly as it found it.
func TestAFailedReservationTakesNothing(t *testing.T) {
	p := newPool(t, 4, 16)

	if err := p.Reserve(1, 32); err != nil { // two blocks
		t.Fatal(err)
	}
	freeBefore := p.FreeBlocks()

	err := p.Reserve(2, 64) // four blocks, only two free
	if !errors.Is(err, ErrFull) {
		t.Fatalf("an impossible reservation returned %v, want ErrFull", err)
	}
	if got := p.FreeBlocks(); got != freeBefore {
		t.Errorf("a failed reservation left %d blocks free, was %d: it took some on the way out",
			got, freeBefore)
	}
	if len(p.Table(2)) != 0 {
		t.Errorf("a failed reservation left a table of %d blocks behind", len(p.Table(2)))
	}

	// And the pool is still usable for something that does fit.
	if err := p.Reserve(2, 32); err != nil {
		t.Errorf("the pool would not serve a reservation that fits: %v", err)
	}
}

func TestReleaseReturnsEverything(t *testing.T) {
	p := newPool(t, 8, 16)

	for id := range uint64(4) {
		if err := p.Reserve(id, 32); err != nil {
			t.Fatal(err)
		}
	}
	if p.FreeBlocks() != 0 {
		t.Fatalf("four sequences of two blocks left %d of 8 free", p.FreeBlocks())
	}

	for id := range uint64(4) {
		p.Release(id)
	}
	if p.FreeBlocks() != 8 {
		t.Errorf("releasing everything left %d of 8 free", p.FreeBlocks())
	}
	if p.Waste() != 0 {
		t.Errorf("an empty pool reports %v waste", p.Waste())
	}

	// Releasing something that was never held is not an error: a sequence that
	// failed before reserving still gets released on its way out, and making
	// that a special case at every call site is how a leak gets written.
	p.Release(999)
	if p.FreeBlocks() != 8 {
		t.Errorf("releasing an unknown id changed the pool to %d free", p.FreeBlocks())
	}
}

// TestNoBlockIsHandedToTwoSequences is the corruption test. A double-handed
// block is one sequence reading another's keys, which produces fluent text from
// the wrong context and no error anywhere.
func TestNoBlockIsHandedToTwoSequences(t *testing.T) {
	p := newPool(t, 64, 8)
	rng := rand.New(rand.NewPCG(7, 11))

	live := map[uint64]int{}
	for step := range 4000 {
		id := uint64(rng.IntN(20))

		switch {
		case live[id] > 0 && rng.IntN(4) == 0:
			p.Release(id)
			delete(live, id)
		default:
			want := live[id] + 1 + rng.IntN(3)
			if err := p.Reserve(id, want); err == nil {
				live[id] = want
			}
		}

		// Every block appears at most once across every table, and never in a
		// table and the free list at the same time.
		seen := map[int]uint64{}
		for holder := range live {
			for _, block := range p.Table(holder) {
				if other, taken := seen[block]; taken {
					t.Fatalf("step %d: block %d is held by both %d and %d",
						step, block, other, holder)
				}
				seen[block] = holder
			}
		}
		for _, block := range p.freeList() {
			if holder, taken := seen[block]; taken {
				t.Fatalf("step %d: block %d is free and held by %d at once", step, block, holder)
			}
			seen[block] = math.MaxUint64
		}
		if len(seen) != p.Blocks() {
			t.Fatalf("step %d: %d blocks accounted for, pool has %d", step, len(seen), p.Blocks())
		}
	}
}

// TestWasteIsTheInternalFragmentation pins the number the block size is chosen
// against, because a trade needs both of its sides measured.
func TestWasteIsTheInternalFragmentation(t *testing.T) {
	p := newPool(t, 16, 16)

	// One position in a sixteen-position block: fifteen wasted of sixteen.
	if err := p.Reserve(1, 1); err != nil {
		t.Fatal(err)
	}
	if got, want := p.Waste(), 15.0/16.0; math.Abs(got-want) > 1e-9 {
		t.Errorf("one position in a block wastes %v, want %v", got, want)
	}

	// Exactly a block: nothing wasted.
	p.Release(1)
	if err := p.Reserve(1, 16); err != nil {
		t.Fatal(err)
	}
	if p.Waste() != 0 {
		t.Errorf("a full block wastes %v", p.Waste())
	}

	// Seventeen: two blocks, thirty-two slots, fifteen unused.
	if err := p.Reserve(1, 17); err != nil {
		t.Fatal(err)
	}
	if got, want := p.Waste(), 15.0/32.0; math.Abs(got-want) > 1e-9 {
		t.Errorf("seventeen positions waste %v, want %v", got, want)
	}
}

// TestABiggerBlockWastesMore is the trade itself, and it is what makes the block
// size a measurement rather than a guess.
func TestABiggerBlockWastesMore(t *testing.T) {
	lengths := []int{7, 19, 33, 100, 250, 511}

	var previous float64
	for i, blockSize := range []int{4, 8, 16, 32, 64} {
		p := newPool(t, 4096, blockSize)
		for id, length := range lengths {
			if err := p.Reserve(uint64(id), length); err != nil {
				t.Fatal(err)
			}
		}
		waste := p.Waste()
		t.Logf("block size %3d: %5.1f%% wasted, %d blocks held",
			blockSize, 100*waste, p.Blocks()-p.FreeBlocks())

		if i > 0 && waste < previous {
			t.Errorf("block size %d wasted %v, less than the smaller size's %v",
				blockSize, waste, previous)
		}
		previous = waste
	}
}

func TestNonsenseIsRejected(t *testing.T) {
	if _, err := New(0, 16); err == nil {
		t.Error("a pool of no blocks was accepted")
	}
	if _, err := New(16, 0); err == nil {
		t.Error("a block of no positions was accepted")
	}
	p := newPool(t, 4, 16)
	if err := p.Reserve(1, -1); err == nil {
		t.Error("a negative reservation was accepted")
	}
}

func TestFitsAgreesWithReserve(t *testing.T) {
	p := newPool(t, 4, 16)

	for positions := 1; positions <= 80; positions++ {
		fresh := newPool(t, 4, 16)
		fits := fresh.Fits(positions)
		err := fresh.Reserve(1, positions)
		if fits != (err == nil) {
			t.Errorf("Fits(%d) said %v but Reserve returned %v", positions, fits, err)
		}
	}
	_ = p
}
