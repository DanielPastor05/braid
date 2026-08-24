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
		stepBase  = flag.Duration("mock-step", 8*time.Millisecond, "mock backend: fixed cost of a step")
		stepPer   = flag.Duration("mock-per-seq", 200*time.Microsecond, "mock backend: marginal cost per sequence")
	)
	flag.Parse()

	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))

	be := backend.NewMock()
	be.Base = *stepBase
	be.PerSeq = *stepPer

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
		Handler: api.New(scheduler, log).Routes(),
		// No WriteTimeout: a generation is a long-lived stream and a deadline
		// on the whole response would cut it off mid-sentence. The request
		// context, MaxTokens and MaxWait are what bound a request here.
		ReadHeaderTimeout: 5 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	go func() {
		log.Info("listening",
			"addr", *addr, "max_batch", *maxBatch, "queue", *queue, "backend", "mock")
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
