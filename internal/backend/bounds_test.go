package backend

import (
	"bytes"
	"context"
	"io"
	"log/slog"
	"os"
	"strings"
	"sync"
	"testing"
	"time"
)

// recorder captures the server's log, which is where the worker's stderr ends
// up. A frame refused on its header is refused while this side is still writing
// the rest of it, so the reply never arrives and the broken pipe is all the
// caller sees -- the reason lives in the log or nowhere.
type recorder struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (r *recorder) Write(p []byte) (int, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.buf.Write(p)
}

func (r *recorder) String() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.buf.String()
}

// TestTheWorkerRefusesAFrameAboveItsBounds is the request side of the pipe
// getting the cap the reply side got from fuzzing.
//
// The worker reads the row count as a uint32 and sizes five arrays from it, and
// reads slot indices the same way and subscripts the pool with them. Neither had
// a bound. Nothing but this server writes to that pipe, so this is not an attack
// surface -- it is that a scheduler bug arrived as a multi-gigabyte allocation
// with no message rather than as a refusal naming the number, and the free-list
// bug fixed the same afternoon is what a scheduler getting its slot bookkeeping
// wrong actually looks like.
//
// Refusing rather than clamping is the choice worth defending: a row index the
// server should never have sent is a bug in the server, and quietly serving it
// is how the free-list bug survived. The worker dies, the pool fails over, and
// the reason is in the log.
func TestTheWorkerRefusesAFrameAboveItsBounds(t *testing.T) {
	exe := os.Getenv("BRAID_WORKER")
	model := os.Getenv("BRAID_MODEL")
	if exe == "" || model == "" {
		t.Skip("set BRAID_WORKER and BRAID_MODEL to run this against the engine")
	}
	quiet := slog.New(slog.NewTextHandler(io.Discard, nil))

	t.Run("more rows than it was started for", func(t *testing.T) {
		log := &recorder{}
		w, err := NewWorker(exe, model, WorkerOptions{MaxBatch: 2},
			slog.New(slog.NewTextHandler(log, nil)))
		if err != nil {
			t.Fatalf("starting the worker: %v", err)
		}
		defer w.Close()

		three := [][]int32{oneWindow(1), oneWindow(2), oneWindow(3)}
		_, err = w.Step(context.Background(), three,
			[]int32{1, 1, 1}, []int32{-1, -1, -1},
			[]float32{1, 1, 1}, []uint64{1, 2, 3})
		if err == nil {
			t.Fatal("a frame of three rows was accepted by a worker started for two")
		}
		t.Logf("the caller saw: %v", err)

		// The forwarding is another goroutine, so give it a moment to catch up.
		deadline := time.Now().Add(2 * time.Second)
		for !strings.Contains(log.String(), "3 rows") && time.Now().Before(deadline) {
			time.Sleep(10 * time.Millisecond)
		}
		if !strings.Contains(log.String(), "3 rows") {
			t.Errorf("the reason never reached the log; it holds: %s", log.String())
		}
	})

	t.Run("a slot above the pool it allocated", func(t *testing.T) {
		w, err := NewWorker(exe, model, WorkerOptions{
			Cache: true, CacheSlots: 2, MaxBatch: 8,
		}, quiet)
		if err != nil {
			t.Fatalf("starting the worker: %v", err)
		}
		defer w.Close()

		// Two rows, because the cached path needs more than one, and one of them
		// names a slot the pool does not have.
		two := [][]int32{oneWindow(1), oneWindow(2)}
		_, err = w.Step(context.Background(), two,
			[]int32{1, 1}, []int32{0, 7},
			[]float32{1, 1}, []uint64{1, 2})
		if err == nil {
			t.Fatal("slot 7 was accepted by a worker that allocated two slots")
		}
		if !strings.Contains(err.Error(), "slot") {
			t.Errorf("the refusal does not say what was wrong: %v", err)
		}
		t.Logf("refused: %v", err)
	})

	// And the bounds do not refuse what is inside them, which is the half a
	// check like this usually gets wrong.
	t.Run("and it still serves a frame within them", func(t *testing.T) {
		w, err := NewWorker(exe, model, WorkerOptions{
			Cache: true, CacheSlots: 4, MaxBatch: 4,
		}, quiet)
		if err != nil {
			t.Fatalf("starting the worker: %v", err)
		}
		defer w.Close()

		two := [][]int32{oneWindow(1), oneWindow(2)}
		ids, err := w.Step(context.Background(), two,
			[]int32{1, 1}, []int32{0, 3},
			[]float32{1, 1}, []uint64{1, 2})
		if err != nil {
			t.Fatalf("a frame inside the bounds was refused: %v", err)
		}
		if len(ids) != 2 {
			t.Fatalf("got %d ids for two rows", len(ids))
		}
	})
}
