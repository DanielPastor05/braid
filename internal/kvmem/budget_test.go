package kvmem

import (
	"strings"
	"testing"
)

// braid's geometry, so the arithmetic in the package comment is checked rather
// than asserted: 6 heads of 64, 6 blocks, a 1024-id context.
func braidBudget(gigabytes float64, blockSize int) Budget {
	return Budget{
		Heads:     6,
		HeadDim:   64,
		Layers:    6,
		Context:   1024,
		BlockSize: blockSize,
		Bytes:     int64(gigabytes * 1024 * 1024 * 1024),
	}
}

// TestTheArithmeticInTheReadme pins the number the whole memory argument rests
// on. If the geometry changes, this is what says so rather than a stale sentence.
func TestTheArithmeticInTheReadme(t *testing.T) {
	b := braidBudget(4, 16)

	// One sequence at the full context: 6 heads * 64 dim * 1024 positions * 4
	// bytes * 6 layers * 2 tensors.
	const want = 6 * 64 * 1024 * 4 * 6 * 2
	if got := b.BytesPerSequence(); got != want {
		t.Errorf("a sequence at full context costs %d bytes, want %d", got, want)
	}
	t.Logf("one sequence at 1024 positions: %.1f MB", float64(want)/(1024*1024))

	// Sixty-four of them is the batch size the server was measured at.
	sixtyFour := float64(want*64) / (1024 * 1024 * 1024)
	t.Logf("sixty-four sequences: %.2f GB", sixtyFour)
	if sixtyFour < 1.0 || sixtyFour > 1.5 {
		t.Errorf("sixty-four sequences at full context is %.2f GB; the README says about 1.2",
			sixtyFour)
	}
}

func TestABudgetBuysWhatItCanAfford(t *testing.T) {
	b := braidBudget(4, 16)

	blocks, err := b.Blocks()
	if err != nil {
		t.Fatal(err)
	}
	perBlock := b.BytesPerBlock()
	if int64(blocks)*perBlock > b.Bytes {
		t.Errorf("%d blocks of %d bytes is %d, over a budget of %d",
			blocks, perBlock, int64(blocks)*perBlock, b.Bytes)
	}
	if int64(blocks+1)*perBlock <= b.Bytes {
		t.Errorf("%d blocks left room for another; the budget is not being spent", blocks)
	}
	t.Logf("4 GB buys %d blocks of %d positions: %d sequences at full context",
		blocks, b.BlockSize, b.Sequences())
}

// TestABudgetTooSmallForOneSequenceIsRefused is the deadlock that would
// otherwise be discovered under load: a sequence that grows past what is left
// has nothing to evict but itself.
func TestABudgetTooSmallForOneSequenceIsRefused(t *testing.T) {
	b := braidBudget(0.05, 16) // 51 MB against 9 MB a sequence... but only just

	small := b
	small.Bytes = b.BytesPerSequence() - 1
	_, err := small.Blocks()
	if err == nil {
		t.Fatal("a budget too small for one sequence at full context was accepted")
	}
	if !strings.Contains(err.Error(), "full") {
		t.Errorf("the refusal does not explain itself: %v", err)
	}

	// One sequence exactly is enough to be legal, if useless.
	exact := b
	exact.Bytes = b.BytesPerSequence()
	if _, err := exact.Blocks(); err != nil {
		t.Errorf("a budget of exactly one sequence was refused: %v", err)
	}
}

func TestNonsenseBudgetsAreRefused(t *testing.T) {
	for name, b := range map[string]Budget{
		"no heads":   {Heads: 0, HeadDim: 64, Layers: 6, Context: 1024, BlockSize: 16, Bytes: 1 << 30},
		"no layers":  {Heads: 6, HeadDim: 64, Layers: 0, Context: 1024, BlockSize: 16, Bytes: 1 << 30},
		"no context": {Heads: 6, HeadDim: 64, Layers: 6, Context: 0, BlockSize: 16, Bytes: 1 << 30},
		"no bytes":   {Heads: 6, HeadDim: 64, Layers: 6, Context: 1024, BlockSize: 16, Bytes: 0},
	} {
		if _, err := b.Blocks(); err == nil {
			t.Errorf("%s was accepted", name)
		}
	}
}
