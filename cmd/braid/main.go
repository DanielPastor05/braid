// Command braid serves the character-level language model from cpp-ai-engine
// behind one continuously batched scheduler.
//
// -worker names the process that holds the model, and -workers how many of them
// to run: more than one is redundancy rather than capacity, because the
// scheduler advances the batch one step at a time and only one worker computes
// at a time.
//
// Without -worker it runs against the mock backend, which produces
// deterministic nonsense at a configurable cost per step. That is enough to
// exercise the scheduler, the HTTP surface and the load harness, and it is
// labelled everywhere it appears so that no number taken from it is ever
// mistaken for a number about the model.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"time"

	"github.com/DanielPastor05/braid/internal/api"
	"github.com/DanielPastor05/braid/internal/backend"
	"github.com/DanielPastor05/braid/internal/sched"
)

func main() {
	var (
		addr = flag.String("addr", ":8080", "address to listen on")
		// One source for the three the scheduler owns, so a swept default lands
		// here without a second place to remember. See sched.Default.
		defaults  = sched.Default()
		maxBatch  = flag.Int("max-batch", defaults.MaxBatch, "most sequences in one forward pass")
		queue     = flag.Int("queue", defaults.QueueDepth, "how many requests may wait for admission")
		maxTokens = flag.Int("max-tokens", defaults.MaxTokensLimit, "longest generation a caller may ask for")
		worker    = flag.String("worker", "", "path to braid_worker; empty runs the mock backend")
		model     = flag.String("model", "models/charlm", "checkpoint prefix the worker loads")
		workers   = flag.Int("workers", 1, "how many worker processes to run behind the scheduler")
		// The engine's own default is 2^22, which keeps a batch of one entirely
		// on the CPU at this model's size: zero kernels launched. 2^20 was
		// measured, not guessed -- it is where the gain flattens out, and it
		// makes single-client serving 3.5x faster. Pass 0 to defer to whatever
		// the engine was built with.
		minFlops = flag.Uint64("cuda-min-flops", 1<<20,
			"engine threshold: matmuls below this many FLOPs stay on the CPU (0 keeps the engine's default)")
		// Left alone. The engine's elementwise default is already 2^20, and the
		// sweep that moved the matmul threshold showed nothing to gain from
		// moving this one -- so it stays where it is rather than being set to
		// its own value and implying a change that was never made.
		minElems = flag.Uint64("cuda-min-elements", 0,
			"engine threshold: elementwise ops below this many elements stay on the CPU (0 keeps the default)")
		// Inert at this model's size and kept for the smaller one. The engine's
		// floor is 2^15 elements; a LayerNorm here sees n*98304 and clears it at
		// a batch of one. It mattered enormously when the model was 96 wide over
		// a 64-id context -- that is the story in the README, and it is history.
		minLayerNorm = flag.Uint64("cuda-min-layernorm", 0,
			"engine threshold: LayerNorms below this many elements stay on the CPU (0 keeps the default)")
		stepBase = flag.Duration("mock-step", 8*time.Millisecond, "mock backend: fixed cost of a step")
		stepPer  = flag.Duration("mock-per-seq", 200*time.Microsecond, "mock backend: marginal cost per sequence")
	)
	flag.Parse()

	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))

	// No silent fallback. A server that quietly runs the mock when the worker
	// fails to start would answer every request with plausible nonsense and
	// report perfectly good latencies for it, and somebody would eventually
	// paste those numbers somewhere. If a worker was asked for and cannot be
	// had, that is a startup failure.
	var (
		be   backend.Backend
		kind string
	)
	if *worker != "" {
		opts := backend.WorkerOptions{
			MinMatmulFlops:       *minFlops,
			MinElements:          *minElems,
			MinLayerNormElements: *minLayerNorm,
		}
		// One worker is a Worker rather than a Pool of one. The pool's failover
		// has nowhere to go with a single process, and a plain worker makes that
		// visible in the log line instead of implying a redundancy that is not
		// there.
		if *workers > 1 {
			p, err := backend.NewPool(*worker, *model, *workers, opts, log)
			if err != nil {
				log.Error("the pool would not start", "error", err, "worker", *worker, "model", *model)
				os.Exit(1)
			}
			be, kind = p, fmt.Sprintf("pool of %d", *workers)
		} else {
			w, err := backend.NewWorker(*worker, *model, opts, log)
			if err != nil {
				log.Error("the worker would not start", "error", err, "worker", *worker, "model", *model)
				os.Exit(1)
			}
			be, kind = w, "worker"
		}
	} else {
		mock := backend.NewMock()
		mock.Base = *stepBase
		mock.PerSeq = *stepPer
		be, kind = mock, "mock"
		log.Warn("running the mock backend: every number this produces is about the scheduler, not a model")
	}

	scheduler, err := sched.New(be, sched.Config{
		MaxBatch:       *maxBatch,
		QueueDepth:     *queue,
		MaxTokensLimit: *maxTokens,
	})
	if err != nil {
		log.Error("the scheduler would not start", "error", err)
		os.Exit(1)
	}

	srv := &http.Server{
		Addr:    *addr,
		Handler: api.New(scheduler, be, log).Routes(),
		// No WriteTimeout: a generation is a long-lived stream and a deadline
		// on the whole response would cut it off mid-sentence. The request
		// context, MaxTokens and MaxWait are what bound a request here.
		ReadHeaderTimeout: 5 * time.Second,

		// A connection that opens and then says nothing costs a goroutine and a
		// file descriptor for as long as it likes. With no WriteTimeout, this
		// and ReadHeaderTimeout are the whole defence.
		//
		// ReadTimeout is deliberately not set, and it is the tempting one to
		// add. It would bound the body as well as the headers, but net/http
		// keeps a background read on the connection to notice a client going
		// away, and that read shares the deadline: once it expires the server
		// treats the connection as gone and cancels the request. A generation
		// still streaming happily would be cut off for the crime of lasting
		// longer than a timeout meant for a request nobody was sending.
		IdleTimeout: 60 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	go func() {
		log.Info("listening",
			"addr", *addr, "max_batch", *maxBatch, "queue", *queue, "backend", kind)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Error("the listener stopped", "error", err)
			stop()
		}
	}()

	<-ctx.Done()
	log.Info("shutting down")

	shutdown, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdown); err != nil {
		log.Error("shutdown did not finish cleanly", "error", err)
	}
	if err := scheduler.Close(); err != nil {
		log.Error("the scheduler did not close cleanly", "error", err)
	}
	log.Info("stopped", "stats", scheduler.Stats())
}
