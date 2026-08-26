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

// TestAWorkerDyingUnderLoadCostsNobodyTheirRequest kills a worker process, with
// the operating system rather than politely, while generations are in flight.
//
// A signal would exercise the shutdown path, which is not the path in question.
// This takes the process away mid-step, so the server discovers it as a broken
// pipe on a write it had every reason to expect to succeed -- which is what
// actually happens when a worker is killed by the OOM killer, by a driver reset,
// or by somebody's `kill -9`.
//
// The property being tested is that no caller loses anything. Not that the
// server survives, which is easy, but that the requests the dead worker was
// halfway through finish normally on another one, with their token streams
// intact. That is possible only because a worker holds no state between steps:
// the scheduler keeps the history and sends the whole window every time, so a
// retry is the same bytes to a different process.
//
// Skipped unless BRAID_WORKER and BRAID_MODEL are set, which CI has neither of.
func TestAWorkerDyingUnderLoadCostsNobodyTheirRequest(t *testing.T) {
	exe := os.Getenv("BRAID_WORKER")
	model := os.Getenv("BRAID_MODEL")
	if exe == "" || model == "" {
		t.Skip("set BRAID_WORKER and BRAID_MODEL to run this against the engine")
	}
	if _, err := os.Stat(exe); err != nil {
		t.Fatalf("BRAID_WORKER is set to %s, which cannot be opened: %v", exe, err)
	}

	quiet := slog.New(slog.NewTextHandler(io.Discard, nil))
	pool, err := backend.NewPool(exe, model, 3, backend.WorkerOptions{
		MinMatmulFlops:       1 << 20,
		MinLayerNormElements: 2048,
	}, quiet)
	if err != nil {
		t.Fatalf("starting the pool: %v", err)
	}

	s, err := New(pool, Config{MaxBatch: 8, QueueDepth: 64, MaxTokensLimit: 1024})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	const clients = 8
	const tokens = 400

	type outcome struct {
		text string
		res  Result
	}
	results := make([]outcome, clients)

	var wg sync.WaitGroup
	for i := range clients {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			text, res := run(t, s, Request{
				Prompt:      "The engine ",
				MaxTokens:   tokens,
				Temperature: 0.7,
				Seed:        uint64(2000 + i),
			})
			results[i] = outcome{text, res}
		}(i)
	}

	// Long enough in that every client is generating and the pool has spread
	// itself over all three workers, short enough that there is plenty of work
	// left to fail.
	time.Sleep(250 * time.Millisecond)

	pids := pool.Pids()
	if len(pids) < 2 {
		t.Fatalf("expected a pool of three live workers, found %d", len(pids))
	}
	victim, err := os.FindProcess(pids[0])
	if err != nil {
		t.Fatalf("finding worker %d: %v", pids[0], err)
	}
	if err := victim.Kill(); err != nil {
		t.Fatalf("killing worker %d: %v", pids[0], err)
	}
	t.Logf("killed worker pid %d with %d clients in flight", pids[0], clients)

	wg.Wait()

	for i, got := range results {
		if got.res.Err != nil {
			t.Errorf("client %d lost its request to a worker it never knew about: %v", i, got.res.Err)
			continue
		}
		if len(got.text) != tokens {
			t.Errorf("client %d got %d characters, wanted %d", i, len(got.text), tokens)
		}
	}

	stats := pool.PoolStats()
	t.Logf("pool: %d deaths, %d failovers, %d restarts, %d/%d live at the end",
		stats.Deaths, stats.Failovers, stats.Restarts, stats.Live, stats.Workers)

	// The test is only worth anything if the kill actually landed while work was
	// in flight. A pass with nothing noticed means the sleep was wrong, not that
	// failover works.
	if stats.Deaths == 0 {
		t.Fatal("no worker death was recorded: the kill missed, and this test proved nothing")
	}
	if stats.Failovers == 0 {
		t.Fatal("a worker died but no step ever failed over, so the death fell between steps " +
			"and the retry path was never taken")
	}

	// And the replacement has to actually come back, or the pool degrades to one
	// worker every time this happens.
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if pool.PoolStats().Live == 3 {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if live := pool.PoolStats().Live; live != 3 {
		t.Errorf("the pool did not refill: %d live workers ten seconds after the death", live)
	}
}

// TestAPoolOfOneStillServes is the degenerate case, and it is here because the
// failover path must not be the only path that works. With one worker there is
// nobody to fail over to, and the pool has to behave exactly like the single
// worker it wraps.
func TestAPoolOfOneStillServes(t *testing.T) {
	exe := os.Getenv("BRAID_WORKER")
	model := os.Getenv("BRAID_MODEL")
	if exe == "" || model == "" {
		t.Skip("set BRAID_WORKER and BRAID_MODEL to run this against the engine")
	}

	quiet := slog.New(slog.NewTextHandler(io.Discard, nil))
	pool, err := backend.NewPool(exe, model, 1, backend.WorkerOptions{
		MinMatmulFlops:       1 << 20,
		MinLayerNormElements: 2048,
	}, quiet)
	if err != nil {
		t.Fatalf("starting the pool: %v", err)
	}

	s, err := New(pool, Config{MaxBatch: 8, QueueDepth: 32, MaxTokensLimit: 1024})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	text, res := run(t, s, Request{
		Prompt: "The engine ", MaxTokens: 120, Temperature: 0.7, Seed: 99,
	})
	if res.Err != nil {
		t.Fatalf("a pool of one failed to serve: %v", res.Err)
	}
	if len(text) != 120 {
		t.Errorf("got %d characters, wanted 120", len(text))
	}
	if stats := pool.PoolStats(); stats.Deaths != 0 || stats.Failovers != 0 {
		t.Errorf("a quiet run recorded %d deaths and %d failovers", stats.Deaths, stats.Failovers)
	}
}
