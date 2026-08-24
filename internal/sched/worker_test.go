package sched

import (
	"io"
	"log/slog"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/DanielPastor05/braid/internal/backend"
)

// TestBatchingDoesNotChangeOutputOnTheRealModel is the same property as
// TestBatchingDoesNotChangeOutput, against the engine instead of the mock.
//
// It is a separate test because it can fail for a reason the mock cannot
// produce. The mock's next token is a hash, so it is bit-exact by construction.
// The real path samples from logits that come off a GPU, and a matrix product
// tiled for a batch of one need not reduce in the same order as the same
// product tiled for a batch of thirty-two. Two logits within a float of each
// other could then land on different sides of the sampling threshold, and the
// sequence would diverge -- not from a scheduler bug, but from arithmetic.
//
// Whether that actually happens on this model and this card is a question about
// the world, not about the code, so it is measured here rather than assumed
// either way.
//
// Skipped unless BRAID_WORKER and BRAID_MODEL point at a built worker and a
// trained checkpoint, which CI has neither of.
func TestBatchingDoesNotChangeOutputOnTheRealModel(t *testing.T) {
	exe := os.Getenv("BRAID_WORKER")
	model := os.Getenv("BRAID_MODEL")
	if exe == "" || model == "" {
		t.Skip("set BRAID_WORKER and BRAID_MODEL to run this against the engine")
	}
	// Set but wrong is a mistake, not a reason to skip. An earlier version
	// skipped on a bad path and reported a green run for a test that never
	// started, which is the failure mode this whole file exists to avoid.
	if _, err := os.Stat(exe); err != nil {
		t.Fatalf("BRAID_WORKER is set to %s, which cannot be opened: %v", exe, err)
	}
	if _, err := os.Stat(model + ".bin"); err != nil {
		t.Fatalf("BRAID_MODEL is set to %s, but %s.bin cannot be opened: %v", model, model, err)
	}

	quiet := slog.New(slog.NewTextHandler(io.Discard, nil))
	subject := Request{Prompt: "The engine ", MaxTokens: 120, Temperature: 0.7, Seed: 99}

	// Alone.
	solo, err := backend.NewWorker(exe, model, quiet)
	if err != nil {
		t.Fatalf("starting the worker: %v", err)
	}
	one, err := New(solo, Config{MaxBatch: 32, QueueDepth: 64, MaxTokensLimit: 1024})
	if err != nil {
		t.Fatal(err)
	}
	want, res := run(t, one, subject)
	if res.Err != nil {
		t.Fatalf("the run on its own failed: %v", res.Err)
	}
	if err := one.Close(); err != nil {
		t.Fatal(err)
	}

	// In company.
	shared, err := backend.NewWorker(exe, model, quiet)
	if err != nil {
		t.Fatalf("starting the worker: %v", err)
	}
	many, err := New(shared, Config{MaxBatch: 32, QueueDepth: 64, MaxTokensLimit: 1024})
	if err != nil {
		t.Fatal(err)
	}
	defer many.Close()

	var wg sync.WaitGroup
	for i := range 15 {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			time.Sleep(time.Duration(i) * time.Millisecond)
			run(t, many, Request{
				Prompt:      "noise",
				MaxTokens:   20 + i*6,
				Temperature: 0.9,
				Seed:        uint64(500 + i),
			})
		}(i)
	}

	got, res := run(t, many, subject)
	wg.Wait()

	if res.Err != nil {
		t.Fatalf("the run in company failed: %v", res.Err)
	}
	snap := many.Stats()
	if snap.MeanBatch <= 1.0 {
		t.Fatalf("mean batch %.2f: the sequences never shared a step, so nothing was tested",
			snap.MeanBatch)
	}
	// Logged rather than only asserted, because "identical output while batched"
	// is a claim about a number, and the number belongs next to the claim.
	t.Logf("%d characters compared at mean batch %.2f over %d steps",
		len(want), snap.MeanBatch, snap.Steps)
	if got != want {
		t.Errorf("sharing a batch changed the model's output at mean batch %.2f.\n alone: %q\nshared: %q",
			snap.MeanBatch, want, got)
	}
}

// TestWorkerRoundTripsTheAlphabet checks the one thing the pipe protocol cannot
// check for itself: that this process and the worker agree on what a token id
// means. A disagreement here produces fluent-looking text made of the wrong
// characters, which no error path would ever catch.
func TestWorkerRoundTripsTheAlphabet(t *testing.T) {
	exe := os.Getenv("BRAID_WORKER")
	model := os.Getenv("BRAID_MODEL")
	if exe == "" || model == "" {
		t.Skip("set BRAID_WORKER and BRAID_MODEL to run this against the engine")
	}

	w, err := backend.NewWorker(exe, model, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err != nil {
		t.Fatalf("starting the worker: %v", err)
	}
	defer w.Close()

	const text = "The engine has a buffer."
	if got := w.Decode(w.Encode(text)); got != text {
		t.Errorf("the alphabet did not round trip: %q became %q", text, got)
	}

	// A window of the model's width, so the shape is the one Step demands.
	window := make([]int32, w.SeqLen())
	copy(window[w.SeqLen()-4:], w.Encode("The "))

	out, err := w.Step([][]int32{window}, []float32{0.7}, []uint64{1})
	if err != nil {
		t.Fatalf("step: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("asked for one sequence, got %d ids back", len(out))
	}
	if out[0] < 0 || int(out[0]) >= w.VocabSize() {
		t.Errorf("id %d is outside a %d-symbol alphabet", out[0], w.VocabSize())
	}

	// And the same window twice must give the same id: sampling is seeded per
	// sequence, so a repeat is not a coin flip.
	again, err := w.Step([][]int32{window}, []float32{0.7}, []uint64{1})
	if err != nil {
		t.Fatalf("step: %v", err)
	}
	if again[0] != out[0] {
		t.Errorf("the same window and seed gave %d then %d", out[0], again[0])
	}

}
