package backend

import (
	"context"
	"fmt"
	"math/rand/v2"
	"os"
	"strconv"
	"testing"
)

// TestBatchInvarianceDivergenceRate measures the thing the other invariance
// tests assert.
//
// Those run one request alone, run it again in company, and compare the text.
// One comparison, one seed, one batch size: a single sample of a property that
// is stochastic by nature, and a pass says only that it did not happen that
// time. This asks how often it happens instead.
//
// The mechanism it is looking for is not a scheduler bug -- the scheduler
// cannot mix rows, and TestBatchingDoesNotChangeOutput holds it to that on a
// backend where the answer is a hash. It is arithmetic. A batch of n is a
// different matrix product from a batch of one, the engine picks its matmul
// kernel from `rows = n * 64` with a cut at 128, and the sampler walks an
// inverse CDF: a last-bit difference near a boundary flips the token, and from
// there two identical requests part.
//
// So: the same window and the same seed, sampled alone and then as one row of a
// batch of n, many times, counting disagreements. The number that comes out is
// what the README is entitled to claim.
//
// BRAID_DIVERGENCE_TRIALS sets the trials per batch size; the default keeps the
// test to a few seconds and is far too small to quote. The figure in the README
// was taken at a much larger one, which the log line records.
func TestBatchInvarianceDivergenceRate(t *testing.T) {
	exe := os.Getenv("BRAID_WORKER")
	model := os.Getenv("BRAID_MODEL")
	if exe == "" || model == "" {
		t.Skip("set BRAID_WORKER and BRAID_MODEL to run this against the engine")
	}

	trials := 200
	if raw := os.Getenv("BRAID_DIVERGENCE_TRIALS"); raw != "" {
		parsed, err := strconv.Atoi(raw)
		if err != nil || parsed < 1 {
			t.Fatalf("BRAID_DIVERGENCE_TRIALS=%q is not a positive number", raw)
		}
		trials = parsed
	}

	w, err := NewWorker(exe, model, WorkerOptions{
		MinMatmulFlops:       1 << 20,
		MinLayerNormElements: 2048,
	}, quiet())
	if err != nil {
		t.Fatalf("starting the worker: %v", err)
	}
	defer w.Close()

	ctx := context.Background()
	vocab := int32(w.VocabSize())
	sizes := []int{2, 4, 8, 16, 32}

	t.Logf("%d trials per batch size, %d-symbol alphabet", trials, vocab)
	t.Logf("%6s | %10s | %s", "batch", "diverged", "rate")

	var worst float64
	for _, n := range sizes {
		// One generator per batch size, seeded from the size, so a rerun of the
		// same size draws the same windows and the number is comparable.
		rng := rand.New(rand.NewPCG(uint64(n), 0x9E3779B97F4A7C15))

		// A row of random ids, right-padded past its length the way the
		// scheduler pads. The length is drawn too: rows of different lengths in
		// one batch sample at different positions, which is the arrangement
		// that did not exist before the padding moved.
		row := func() ([]int32, int32) {
			length := 1 + rng.IntN(w.SeqLen())
			window := make([]int32, w.SeqLen())
			for i := range length {
				window[i] = int32(rng.IntN(int(vocab)))
			}
			return window, int32(length)
		}

		var diverged int
		for trial := range trials {
			window, length := row()
			seed := rng.Uint64()
			temp := float32(0.5 + rng.Float64())

			alone, err := w.Step(ctx, [][]int32{window}, []int32{length},
				[]float32{temp}, []uint64{seed})
			if err != nil {
				t.Fatalf("batch 1, trial %d: %v", trial, err)
			}

			// The same row, at a position that moves, among neighbours that are
			// different every time. A fixed position or fixed filler would test
			// one arrangement of the batch rather than the batch.
			at := rng.IntN(n)
			windows := make([][]int32, n)
			lengths := make([]int32, n)
			temps := make([]float32, n)
			seeds := make([]uint64, n)
			for i := range windows {
				if i == at {
					windows[i], lengths[i] = window, length
					temps[i], seeds[i] = temp, seed
					continue
				}
				windows[i], lengths[i] = row()
				temps[i] = float32(0.5 + rng.Float64())
				seeds[i] = rng.Uint64()
			}

			together, err := w.Step(ctx, windows, lengths, temps, seeds)
			if err != nil {
				t.Fatalf("batch %d, trial %d: %v", n, trial, err)
			}
			if together[at] != alone[0] {
				diverged++
			}
		}

		rate := float64(diverged) / float64(trials)
		if rate > worst {
			worst = rate
		}
		t.Logf("%6d | %10s | %.4f%%", n,
			fmt.Sprintf("%d/%d", diverged, trials), rate*100)
	}

	// The assertion is a ceiling, not a claim of zero. Zero is what has been
	// observed and it is not what is guaranteed -- nothing in the engine
	// promises the same reduction order at two batch sizes, and a test that
	// demanded it would be asserting a property the code does not have. A rate
	// that climbs past a percent means something changed that is worth looking
	// at, and that is what this is for.
	if worst > 0.01 {
		t.Errorf("worst divergence rate %.4f%% across batch sizes: sequences are being "+
			"measurably changed by their neighbours", worst*100)
	}
}
