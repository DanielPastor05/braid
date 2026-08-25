// Command braid serves the character-level language model from cpp-ai-engine
// behind one continuously batched scheduler.
//
// There is no engine wired in yet: this build runs against the mock backend,
// which produces deterministic nonsense at a configurable cost per step. It is
// enough to exercise the scheduler, the HTTP surface and the load harness, and
// it is labelled everywhere it appears so that no number taken from it is ever
// mistaken for a number about the model.
package main

import (
	"context"
	"errors"
	"flag"
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
		addr      = flag.String("addr", ":8080", "address to listen on")
		maxBatch  = flag.Int("max-batch", 32, "most sequences in one forward pass")
		queue     = flag.Int("queue", 256, "how many requests may wait for admission")
		maxTokens = flag.Int("max-tokens", 1024, "longest generation a caller may ask for")
		worker    = flag.String("worker", "", "path to braid_worker; empty runs the mock backend")
		model     = flag.String("model", "models/charlm", "checkpoint prefix the worker loads")
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
		// The engine's default is 2^15, which this model's LayerNorms -- n*6144
		// elements -- do not clear until a batch of six. Below that all four of
		// them refuse and the forward comes home at each one. Any value under
		// 6 144 puts a single sequence on the card; 2 048 leaves room and is
		// what the numbers on this repo were measured at. Pass 0 to defer to
		// whatever the engine was built with.
		minLayerNorm = flag.Uint64("cuda-min-layernorm", 2048,
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
		w, err := backend.NewWorker(*worker, *model, backend.WorkerOptions{
			MinMatmulFlops:       *minFlops,
			MinElements:          *minElems,
			MinLayerNormElements: *minLayerNorm,
		}, log)
		if err != nil {
			log.Error("the worker would not start", "error", err, "worker", *worker, "model", *model)
			os.Exit(1)
		}
		be, kind = w, "worker"
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
