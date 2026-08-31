package backend

import (
	"context"
	"io"
	"log/slog"
	"math"
	"math/rand/v2"
	"os"
	"strconv"
	"testing"
)

// How far apart are the logits, rather than how often does the token change?
//
// TestBatchInvarianceDivergenceRate compares the sampled *token*, and that has a
// blind spot worth naming: the sampler walks an inverse CDF, so a difference in
// the last bits only changes the answer when it happens to fall across a
// boundary. Counting flipped tokens therefore measures the probability that the
// noise mattered, which depends on the shape of the distribution, and not the
// noise. A model with a confident peak would report zero divergences while
// carrying any amount of arithmetic drift underneath.
//
// This measures the drift. Same window, same seed, alone and then as one row of
// a batch of n, comparing every logit of the sampled row.
//
// Both numbers are worth having and they answer different questions. The token
// rate is what a caller experiences. This is what the arithmetic is doing, and
// it is the one that would move first if a change to the engine's dispatch or
// this server's batching started costing precision.
func TestBatchInvarianceLogitDrift(t *testing.T) {
	exe := os.Getenv("BRAID_WORKER")
	model := os.Getenv("BRAID_MODEL")
	if exe == "" || model == "" {
		t.Skip("set BRAID_WORKER and BRAID_MODEL to run this against the engine")
	}

	trials := 40
	if raw := os.Getenv("BRAID_DRIFT_TRIALS"); raw != "" {
		parsed, err := strconv.Atoi(raw)
		if err != nil || parsed < 1 {
			t.Fatalf("BRAID_DRIFT_TRIALS=%q is not a positive number", raw)
		}
		trials = parsed
	}

	w, err := NewWorker(exe, model, WorkerOptions{EmitLogits: true},
		slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err != nil {
		t.Fatalf("starting the worker: %v", err)
	}
	defer w.Close()

	ctx := context.Background()
	vocab := int32(w.VocabSize())
	sizes := []int{2, 4, 8, 16, 32}

	t.Logf("%d trials per batch size, %d-symbol alphabet", trials, vocab)
	t.Logf("%6s | %12s | %12s | %s", "batch", "max abs", "max rel", "tokens differing")

	var worstRelative float64
	for _, n := range sizes {
		rng := rand.New(rand.NewPCG(uint64(n), 0x5DEECE66D))

		row := func() ([]int32, int32) {
			length := 1 + rng.IntN(w.SeqLen())
			window := make([]int32, w.SeqLen())
			for i := range length {
				window[i] = int32(rng.IntN(int(vocab)))
			}
			return window, int32(length)
		}

		var maxAbs, maxRel float64
		differing := 0

		for trial := range trials {
			window, length := row()
			seed := rng.Uint64()
			temp := float32(0.5 + rng.Float64())

			aloneIDs, err := w.Step(ctx, [][]int32{window}, []int32{length},
				[]float32{temp}, []uint64{seed})
			if err != nil {
				t.Fatalf("batch 1, trial %d: %v", trial, err)
			}
			alone := append([]float32(nil), w.LastLogits()...)
			if len(alone) != int(vocab) {
				t.Fatalf("batch 1 returned %d logits for a %d-symbol alphabet; "+
					"EmitLogits did not reach the worker", len(alone), vocab)
			}

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

			togetherIDs, err := w.Step(ctx, windows, lengths, temps, seeds)
			if err != nil {
				t.Fatalf("batch %d, trial %d: %v", n, trial, err)
			}
			together := w.LastLogits()[at*int(vocab) : (at+1)*int(vocab)]

			if togetherIDs[at] != aloneIDs[0] {
				differing++
			}
			for i := range alone {
				diff := math.Abs(float64(alone[i] - together[i]))
				maxAbs = max(maxAbs, diff)
				// Relative to the magnitude, floored at one, so a logit near
				// zero does not turn a tiny absolute difference into a huge
				// ratio and dominate the number.
				maxRel = max(maxRel, diff/max(1, math.Abs(float64(alone[i]))))
			}
		}

		worstRelative = max(worstRelative, maxRel)
		t.Logf("%6d | %12.3e | %12.3e | %d/%d", n, maxAbs, maxRel, differing, trials)
	}

	// A ceiling, not an equality. float32 carries about 1e-7 of relative
	// precision, and two different matmul shapes accumulate differently across
	// six blocks, so some drift is arithmetic and not a bug. What this catches is
	// drift large enough to be a bug: a wrong reduction, a stale buffer, a
	// changed dispatch that quietly rounds worse.
	const ceiling = 1e-3
	if worstRelative > ceiling {
		t.Errorf("logits drift by %.3e relative between a sequence alone and in a batch, "+
			"ceiling %.0e -- that is too far to be rounding", worstRelative, ceiling)
	}
}
