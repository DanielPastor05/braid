package backend

import (
	"os"
	"testing"
	"time"
)

// step drives one sequence through the pool and checks the answer is the one
// that window should produce, whichever worker happened to produce it. That is
// the whole claim: a caller cannot tell which process served it.
func step(t *testing.T, p *Pool, window []int32) error {
	t.Helper()

	out, err := p.Step([][]int32{window}, []float32{0.7}, []uint64{1})
	if err != nil {
		return err
	}
	if want := fakeNextID(fakeWindowBytes(window)); out[0] != want {
		t.Fatalf("the pool answered %d where this window means %d: a failover returned "+
			"somebody else's work", out[0], want)
	}
	return nil
}

// waitForLive gives the refill goroutine time to put a worker back, and says
// what it saw if it never does.
func waitForLive(t *testing.T, p *Pool, want int) {
	t.Helper()

	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if p.PoolStats().Live == want {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Errorf("the pool never got back to %d live workers: %+v", want, p.PoolStats())
}

// TestPoolFailsOverWhenAWorkerIsKilled is the CI half of the chaos test.
//
// The version in internal/sched runs the whole server against the real engine
// and is skipped everywhere without a GPU, which left this -- the mechanism the
// README leads with -- verified on one desk. This kills a process the same way,
// with the operating system rather than a signal, and asserts the thing that
// actually matters: not that the pool survives, but that every step still
// returns the answer its window earns.
func TestPoolFailsOverWhenAWorkerIsKilled(t *testing.T) {
	exe, prefix := startFake(t, "normal")
	pool, err := NewPool(exe, prefix, 3, WorkerOptions{}, quiet())
	if err != nil {
		t.Fatalf("starting the pool: %v", err)
	}
	defer pool.Close()

	// Enough steps that the round robin has visited all three.
	for i := range 6 {
		if err := step(t, pool, oneWindow(int32(i))); err != nil {
			t.Fatalf("step %d before the kill: %v", i, err)
		}
	}
	before := pool.Timings().Steps

	pids := pool.Pids()
	if len(pids) != 3 {
		t.Fatalf("expected three live workers, got %d", len(pids))
	}
	victim, err := os.FindProcess(pids[0])
	if err != nil {
		t.Fatalf("finding worker %d: %v", pids[0], err)
	}
	if err := victim.Kill(); err != nil {
		t.Fatalf("killing worker %d: %v", pids[0], err)
	}

	// Every one of these has to answer correctly. One of them will be handed to
	// a process that is no longer there.
	for i := range 12 {
		if err := step(t, pool, oneWindow(int32(100+i))); err != nil {
			t.Fatalf("step %d after the kill was not served at all: %v", i, err)
		}
	}

	stats := pool.PoolStats()
	if stats.Deaths == 0 {
		t.Fatal("the pool never noticed the death, so nothing here was a failover")
	}
	if stats.Failovers == 0 {
		t.Fatal("a death was recorded but no step failed over, which cannot both be true")
	}
	waitForLive(t, pool, 3)

	// The replaced worker's counters have to survive it. Without the retired
	// accumulator in retire(), a restart would silently reset the step history
	// and every timing the load harness prints would be wrong afterwards.
	if after := pool.Timings().Steps; after <= before {
		t.Errorf("step count went from %d to %d across a restart: the dead worker's "+
			"history was dropped", before, after)
	}
}

// TestPoolWithEveryWorkerDeadReturnsAnError checks the end of the road. Each
// worker dies on its first step, so there is nobody to fail over to, and the
// pool has to say so rather than retry forever or block.
func TestPoolWithEveryWorkerDeadReturnsAnError(t *testing.T) {
	exe, prefix := startFake(t, "die:0")
	pool, err := NewPool(exe, prefix, 3, WorkerOptions{}, quiet())
	if err != nil {
		t.Fatalf("starting the pool: %v", err)
	}
	defer pool.Close()

	done := make(chan error, 1)
	go func() {
		_, err := pool.Step([][]int32{oneWindow(1)}, []float32{0.7}, []uint64{1})
		done <- err
	}()

	select {
	case err := <-done:
		if err == nil {
			t.Fatal("a pool whose every worker died reported a success")
		}
	case <-time.After(30 * time.Second):
		t.Fatal("Step never returned: the pool retried past the end of its own slots")
	}

	if deaths := pool.PoolStats().Deaths; deaths < 3 {
		t.Errorf("expected all three workers to be recorded dead, got %d", deaths)
	}
}

// TestPoolOfOneHasNowhereToFailOver is the degenerate case. The pool must not
// pretend: one worker refusing is the request failing, and the worker's own
// message has to come with it.
func TestPoolOfOneHasNowhereToFailOver(t *testing.T) {
	exe, prefix := startFake(t, "error:0")
	pool, err := NewPool(exe, prefix, 1, WorkerOptions{}, quiet())
	if err != nil {
		t.Fatalf("starting the pool: %v", err)
	}
	defer pool.Close()

	_, err = pool.Step([][]int32{oneWindow(1)}, []float32{0.7}, []uint64{1})
	if err == nil {
		t.Fatal("a pool of one reported a success while its only worker refused")
	}
	if stats := pool.PoolStats(); stats.Failovers != 1 {
		t.Errorf("expected the single attempt to count as one failover, got %d", stats.Failovers)
	}
}

// TestPoolServesTheSameAlphabetAsAWorker guards the copy. Pool builds its own
// encode table rather than delegating, so that it still answers when every
// worker is down -- and a table that disagreed with the workers' would produce
// text made of the wrong characters, which no error path would catch.
func TestPoolServesTheSameAlphabetAsAWorker(t *testing.T) {
	exe, prefix := startFake(t, "normal")

	pool, err := NewPool(exe, prefix, 2, WorkerOptions{}, quiet())
	if err != nil {
		t.Fatalf("starting the pool: %v", err)
	}
	defer pool.Close()

	w, err := NewWorker(exe, prefix, WorkerOptions{}, quiet())
	if err != nil {
		t.Fatalf("starting a worker: %v", err)
	}
	defer w.Close()

	if pool.SeqLen() != w.SeqLen() || pool.VocabSize() != w.VocabSize() {
		t.Fatalf("pool reports %d/%d where a worker reports %d/%d",
			pool.SeqLen(), pool.VocabSize(), w.SeqLen(), w.VocabSize())
	}

	// Every symbol the alphabet has, through both, in both directions.
	all := make([]int32, w.VocabSize())
	for i := range all {
		all[i] = int32(i)
	}
	if pool.Decode(all) != w.Decode(all) {
		t.Error("pool and worker decode the same ids differently")
	}
	text := w.Decode(all)
	if pool.Decode(pool.Encode(text)) != text {
		t.Error("the pool's alphabet does not round trip")
	}
}

// TestPoolNeedsAtLeastOneWorker is the argument check, and it is here because
// NewPool(…, 0, …) would otherwise build a pool whose pick() loops over nothing
// and whose Step returns a confusing error instead of a clear one.
func TestPoolNeedsAtLeastOneWorker(t *testing.T) {
	exe, prefix := startFake(t, "normal")
	if _, err := NewPool(exe, prefix, 0, WorkerOptions{}, quiet()); err == nil {
		t.Error("a pool of zero workers was accepted")
	}
}
