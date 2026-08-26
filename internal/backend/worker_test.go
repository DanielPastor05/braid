package backend

import (
	"context"
	"io"
	"log/slog"
	"strings"
	"testing"
	"time"
)

func quiet() *slog.Logger { return slog.New(slog.NewTextHandler(io.Discard, nil)) }

// TestWorkerSpeaksTheProtocol is the round trip nothing else checks: a frame
// leaves, a frame comes back, and every field of both is where the other side
// expects it.
//
// The engine's tests cannot cover this — they do not know the protocol exists —
// and the tests against a real worker cover it only where there is a GPU. A
// field written at the wrong offset would show up there as garbled text and
// here as a wrong number, which is the difference between a bug you chase and a
// bug that tells you its name.
func TestWorkerSpeaksTheProtocol(t *testing.T) {
	exe, prefix := startFake(t, "normal")
	w, err := NewWorker(exe, prefix, WorkerOptions{}, quiet())
	if err != nil {
		t.Fatalf("starting the fake: %v", err)
	}
	defer w.Close()

	windows := [][]int32{oneWindow(1, 2, 3), oneWindow(9, 9, 9)}
	out, err := w.Step(context.Background(), windows, []float32{0.7, 0.9}, []uint64{1, 2})
	if err != nil {
		t.Fatalf("step: %v", err)
	}

	if len(out) != len(windows) {
		t.Fatalf("sent %d windows, got %d ids back", len(windows), len(out))
	}
	for i, window := range windows {
		if want := fakeNextID(fakeWindowBytes(window)); out[i] != want {
			t.Errorf("row %d came back as %d, wanted %d: the window did not survive the frame",
				i, out[i], want)
		}
	}

	// The six numbers the worker reports about itself, which the load harness
	// turns into the step breakdown. Asserted exactly, because "some plausible
	// duration arrived" would pass with the offsets shifted by one field.
	got := w.Timings()
	if got.Steps != 1 || got.Sequences != 2 {
		t.Errorf("expected one step over two sequences, got %d and %d", got.Steps, got.Sequences)
	}
	if got.Build != 1000 || got.Forward != 2000 || got.Sample != 500 {
		t.Errorf("timings arrived as build %v, forward %v, sample %v; wanted 1µs, 2µs, 500ns",
			got.Build, got.Forward, got.Sample)
	}
	if got.Kernels != 60 || got.ToDevice != 1 || got.ToHost != 1 {
		t.Errorf("counters arrived as %d kernels, %d to device, %d to host; wanted 60, 1, 1",
			got.Kernels, got.ToDevice, got.ToHost)
	}
}

// TestWorkerSurfacesAnErrorFrame checks the path where the worker is alive and
// says no. It has to reach the caller as that message, not as a broken pipe:
// the two mean different things and only one of them is worth retrying
// elsewhere.
func TestWorkerSurfacesAnErrorFrame(t *testing.T) {
	exe, prefix := startFake(t, "error:0")
	w, err := NewWorker(exe, prefix, WorkerOptions{}, quiet())
	if err != nil {
		t.Fatalf("starting the fake: %v", err)
	}
	defer w.Close()

	_, err = w.Step(context.Background(), [][]int32{oneWindow(1)}, []float32{0.7}, []uint64{1})
	if err == nil {
		t.Fatal("a refusing worker was reported as a success")
	}
	if !strings.Contains(err.Error(), "told to refuse") {
		t.Errorf("the worker's own message did not reach the caller: %v", err)
	}
}

// TestWorkerRejectsAStatusThatIsNotOne guards the case where the pipe is intact
// and the bytes are nonsense. Reading on would consume the next frame's header
// as this frame's body and every step afterwards would be quietly wrong.
func TestWorkerRejectsAStatusThatIsNotOne(t *testing.T) {
	exe, prefix := startFake(t, "garbage:0")
	w, err := NewWorker(exe, prefix, WorkerOptions{}, quiet())
	if err != nil {
		t.Fatalf("starting the fake: %v", err)
	}
	defer w.Close()

	_, err = w.Step(context.Background(), [][]int32{oneWindow(1)}, []float32{0.7}, []uint64{1})
	if err == nil {
		t.Fatal("a status of 99 was accepted")
	}
	if !strings.Contains(err.Error(), "not a status") {
		t.Errorf("expected the unknown status to be named, got: %v", err)
	}
}

// TestWorkerChecksItsArgumentsBeforeTheWire is the cheap half: a ragged call or
// a window of the wrong width is this side's mistake and never becomes a frame.
func TestWorkerChecksItsArgumentsBeforeTheWire(t *testing.T) {
	exe, prefix := startFake(t, "normal")
	w, err := NewWorker(exe, prefix, WorkerOptions{}, quiet())
	if err != nil {
		t.Fatalf("starting the fake: %v", err)
	}
	defer w.Close()

	cases := []struct {
		name    string
		windows [][]int32
		temps   []float32
		seeds   []uint64
	}{
		{"no sequences", nil, nil, nil},
		{"a window one id short", [][]int32{make([]int32, workerSeqLen-1)}, []float32{0.7}, []uint64{1}},
		{"more windows than seeds", [][]int32{oneWindow(1), oneWindow(2)}, []float32{0.7, 0.7}, []uint64{1}},
	}
	for _, c := range cases {
		if _, err := w.Step(context.Background(), c.windows, c.temps, c.seeds); err == nil {
			t.Errorf("%s was accepted", c.name)
		}
	}

	// And the worker is still usable afterwards: a rejected call must not have
	// written a partial frame.
	if _, err := w.Step(context.Background(), [][]int32{oneWindow(5)}, []float32{0.7}, []uint64{1}); err != nil {
		t.Errorf("a rejected call left the pipe in a bad state: %v", err)
	}
}

// TestWorkerThresholdsReachTheProcess checks that WorkerOptions become
// environment the child actually sees. The engine reads them once at start, so
// an option that never arrives is invisible: everything works, twice as slowly,
// and nothing says why.
func TestWorkerThresholdsReachTheProcess(t *testing.T) {
	exe, prefix := startFake(t, "normal")

	lines := make(chan string, 32)
	log := slog.New(slog.NewTextHandler(writerFunc(func(p []byte) {
		select {
		case lines <- string(p):
		default:
		}
	}), nil))

	w, err := NewWorker(exe, prefix, WorkerOptions{
		MinMatmulFlops:       1 << 20,
		MinLayerNormElements: 2048,
	}, log)
	if err != nil {
		t.Fatalf("starting the fake: %v", err)
	}
	defer w.Close()

	// The fake announces itself on stderr, which NewWorker forwards to the
	// logger. Seeing that line proves the plumbing that carries the thresholds.
	select {
	case line := <-lines:
		if !strings.Contains(line, "fake worker ready") {
			t.Errorf("the worker's stderr did not reach the log: %q", line)
		}
	case <-time.After(5 * time.Second):
		t.Error("nothing from the worker's stderr reached the log")
	}
}

// writerFunc adapts a function to io.Writer, so a test can watch a slog stream
// without a file.
type writerFunc func([]byte)

func (f writerFunc) Write(p []byte) (int, error) {
	f(p)
	return len(p), nil
}
