package sched

import (
	"io"
	"log/slog"
	"os"
	"strconv"
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
	solo, err := backend.NewWorker(exe, model, backend.WorkerOptions{}, quiet)
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
	shared, err := backend.NewWorker(exe, model, backend.WorkerOptions{}, quiet)
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

// TestTheCPUAndGPUPathsAgree runs one request twice: once with the engine's own
// CUDA threshold, which at this model's size keeps a batch of one entirely on
// the host, and once with the threshold braid actually serves at, which puts the
// same work on the card.
//
// The two runs are therefore not the same arithmetic. They are a CPU
// implementation and a CUDA implementation of the same model, each sampling two
// hundred times in sequence from logits they computed independently. One
// disagreement anywhere in those two hundred draws diverges the text from that
// point on, so the comparison is far stricter than comparing logits would be.
//
// This is the test that made lowering the threshold safe to do by default. It
// is a measurement about one model on one card, not a promise about either.
func TestTheCPUAndGPUPathsAgree(t *testing.T) {
	exe := os.Getenv("BRAID_WORKER")
	model := os.Getenv("BRAID_MODEL")
	if exe == "" || model == "" {
		t.Skip("set BRAID_WORKER and BRAID_MODEL to run this against the engine")
	}

	quiet := slog.New(slog.NewTextHandler(io.Discard, nil))
	req := Request{Prompt: "The engine ", MaxTokens: 200, Temperature: 0.7, Seed: 99}

	generate := func(opts backend.WorkerOptions) string {
		t.Helper()
		w, err := backend.NewWorker(exe, model, opts, quiet)
		if err != nil {
			t.Fatalf("starting the worker: %v", err)
		}
		s, err := New(w, Config{MaxBatch: 1, QueueDepth: 4, MaxTokensLimit: 1024})
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()

		text, res := run(t, s, req)
		if res.Err != nil {
			t.Fatalf("generation failed: %v", res.Err)
		}
		return text
	}

	onHost := generate(backend.WorkerOptions{MinMatmulFlops: 1 << 22, MinElements: 1 << 22})
	onCard := generate(backend.WorkerOptions{MinMatmulFlops: 1 << 20, MinElements: 1 << 20})

	if len(onHost) != req.MaxTokens {
		t.Fatalf("expected %d characters, got %d", req.MaxTokens, len(onHost))
	}
	if onHost != onCard {
		// Where they part matters: byte 3 is a different failure from byte 190.
		at := 0
		for at < len(onHost) && at < len(onCard) && onHost[at] == onCard[at] {
			at++
		}
		t.Errorf("the two paths diverged at character %d of %d.\n host: %q\n card: %q",
			at, req.MaxTokens, onHost, onCard)
	}
}

// TestKernelsByBatchSize drives the worker at exact batch sizes and reports what
// each one costs.
//
// The load sweep cannot answer this. Its rows are labelled by client count, and
// the batch a step actually gets is whatever had arrived by then -- "four
// clients" measured a mean batch of 3.79 over steps ranging from one to four, so
// its kernel count is an average over sizes and cannot say which size the jump
// belongs to. Here n is exactly n, fifty times in a row.
//
// The assertion is a regression guard on the threshold braid serves at: if a
// batch of one ever stops launching kernels again, the default has drifted back
// to the engine's and single-client serving is three and a half times slower
// than it should be. The table is the point, and it goes to the log.
func TestKernelsByBatchSize(t *testing.T) {
	exe := os.Getenv("BRAID_WORKER")
	model := os.Getenv("BRAID_MODEL")
	if exe == "" || model == "" {
		t.Skip("set BRAID_WORKER and BRAID_MODEL to run this against the engine")
	}

	// Overridable so the same table can be taken at several thresholds, which is
	// how the jump between five and six was located: if it is a threshold being
	// crossed, moving the threshold moves the jump.
	threshold := uint64(1 << 20)
	if raw := os.Getenv("BRAID_MIN_FLOPS"); raw != "" {
		parsed, err := strconv.ParseUint(raw, 10, 64)
		if err != nil {
			t.Fatalf("BRAID_MIN_FLOPS=%q is not a number: %v", raw, err)
		}
		threshold = parsed
	}

	elements := uint64(0)
	if raw := os.Getenv("BRAID_MIN_ELEMENTS"); raw != "" {
		parsed, err := strconv.ParseUint(raw, 10, 64)
		if err != nil {
			t.Fatalf("BRAID_MIN_ELEMENTS=%q is not a number: %v", raw, err)
		}
		elements = parsed
	}

	layernorm := uint64(0)
	if raw := os.Getenv("BRAID_MIN_LAYERNORM"); raw != "" {
		parsed, err := strconv.ParseUint(raw, 10, 64)
		if err != nil {
			t.Fatalf("BRAID_MIN_LAYERNORM=%q is not a number: %v", raw, err)
		}
		layernorm = parsed
	}

	w, err := backend.NewWorker(exe, model, backend.WorkerOptions{
		MinMatmulFlops:       threshold,
		MinElements:          elements,
		MinLayerNormElements: layernorm,
	}, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err != nil {
		t.Fatalf("starting the worker: %v", err)
	}
	defer w.Close()

	const repeats = 50
	batch := func(n int) (windows [][]int32, temps []float32, seeds []uint64) {
		for i := range n {
			window := make([]int32, w.SeqLen())
			copy(window[w.SeqLen()-5:], w.Encode("The e"))
			windows = append(windows, window)
			temps = append(temps, 0.7)
			seeds = append(seeds, uint64(i))
		}
		return windows, temps, seeds
	}

	t.Logf("min_matmul_flops %d, min_elements %d, min_layernorm %d", threshold, elements, layernorm)
	t.Logf("%4s | %8s | %10s | %8s | %11s", "n", "kernels", "to_device", "to_host", "forward ms")
	var atOne float64
	for n := 1; n <= 12; n++ {
		windows, temps, seeds := batch(n)

		// Warm, so the first call's allocations are not charged to the sample.
		for range 5 {
			if _, err := w.Step(windows, temps, seeds); err != nil {
				t.Fatalf("step at n=%d: %v", n, err)
			}
		}

		before := w.Timings()
		for range repeats {
			if _, err := w.Step(windows, temps, seeds); err != nil {
				t.Fatalf("step at n=%d: %v", n, err)
			}
		}
		after := w.Timings()

		kernels := float64(after.Kernels-before.Kernels) / repeats
		toDevice := float64(after.ToDevice-before.ToDevice) / repeats
		toHost := float64(after.ToHost-before.ToHost) / repeats
		forward := (after.Forward - before.Forward).Seconds() * 1000 / repeats
		if n == 1 {
			atOne = kernels
		}
		t.Logf("%4d | %8.0f | %10.0f | %8.0f | %11.3f", n, kernels, toDevice, toHost, forward)
	}

	if atOne == 0 && threshold <= 1<<20 {
		t.Error("a batch of one launched no kernels: the CUDA threshold has drifted back " +
			"to the engine's training default, and single-client serving is 3.5x slower for it")
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

	w, err := backend.NewWorker(exe, model, backend.WorkerOptions{}, slog.New(slog.NewTextHandler(io.Discard, nil)))
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
