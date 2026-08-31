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
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
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
		// On by default, and that is a measurement rather than a preference.
		// It was off while the cache cost the tail what it bought in
		// throughput; batching the prefills removed the tail cost, and what is
		// left is 1.98x the tokens at sixty-four clients for a TTFT p50 of 57 ms
		// against 50. Nothing is traded away any more, so nothing is left for an
		// operator to decide.
		cache = flag.Bool("cache", true,
			"keep a key/value cache in the worker, indexed by slot")
		// The README has always said this server should not be exposed to
		// anything. That was a sentence, and a sentence is not a control. These
		// two are the control: without a token the server refuses to listen
		// anywhere but loopback, so exposing it is now a thing you have to mean.
		authToken = flag.String("auth-token", "",
			"bearer token required on every request; without one the server binds loopback only")
		rate = flag.Float64("rate", 0,
			"requests per second allowed per client address, 0 for no limit")
		burst = flag.Int("burst", 16, "how many requests a client may make back to back")

		worker  = flag.String("worker", "", "path to braid_worker; empty runs the mock backend")
		model   = flag.String("model", "models/charlm", "checkpoint prefix the worker loads")
		workers = flag.Int("workers", 1, "how many worker processes to run behind the scheduler")
		// The engine's own default is 2^22, which keeps a batch of one entirely
		// on the CPU at this model's size: zero kernels launched. 2^20 was
		// measured, not guessed -- it is where the gain flattens out, and it
		// makes single-client serving 3.5x faster. Pass 0 to defer to whatever
		// the engine was built with.
		minFlops = flag.Uint64("cuda-min-flops", 1<<20,
			"engine threshold: matmuls below this many FLOPs stay on the CPU (0 keeps the engine's default)")
		// These two were left at the engine's defaults for as long as a step
		// computed all 256 positions, and at that size the defaults were right:
		// a LayerNorm saw n*98304 elements and cleared its 2^15 floor at a batch
		// of one, so moving it changed nothing.
		//
		// Not computing the padding made a step narrow again, and both floors
		// came back to life on the wrong side. A step at a batch of one over the
		// twenty-nine positions this serves at is 11 136 elements, which is
		// under both -- so every elementwise add and every normalisation went
		// back to the host, one PCIe round trip at a time.
		//
		// Swept, three interleaved passes, then confirmed against the server:
		// lowering both is worth 1.66x at one client and 2.09x at two, a wash
		// from four upward, and it makes the kernel count a flat 177 where it
		// was 140 at one client and 175 at thirty-two. Below four rows in the
		// batch it costs about 40% -- a one-token prompt served alone -- which is
		// not a case this serves and is the reason these are flags.
		//
		// Note what this reverses: an earlier sweep on the 172 728-parameter
		// model found going to 1 slightly *worse*. It was, then. The regime
		// changed twice since.
		minElems = flag.Uint64("cuda-min-elements", 1,
			"engine threshold: elementwise ops below this many elements stay on the CPU (0 keeps the default)")
		minLayerNorm = flag.Uint64("cuda-min-layernorm", 1,
			"engine threshold: LayerNorms below this many elements stay on the CPU (0 keeps the default)")
		stepBase = flag.Duration("mock-step", 8*time.Millisecond, "mock backend: fixed cost of a step")
		stepPer  = flag.Duration("mock-per-seq", 200*time.Microsecond, "mock backend: marginal cost per sequence")
	)
	flag.Parse()

	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))

	// Refusing to start beats starting and hoping. An unauthenticated inference
	// server on a reachable interface is a GPU anybody can spend: sixty-four
	// concurrent requests asking for the maximum token count occupy the batch
	// for minutes, from one machine, with no way to say no.
	if *authToken == "" && !loopbackOnly(*addr) {
		log.Error("refusing to listen on a reachable address without -auth-token",
			"addr", *addr,
			"fix", "pass -auth-token, or bind 127.0.0.1 to keep it local")
		os.Exit(1)
	}

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
			Cache:                *cache,
			// A slot per row that can share a step. Sized here rather than in
			// the worker because the scheduler is what decides MaxBatch.
			CacheSlots: *maxBatch,
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

	guard := api.NewGuard(*authToken, *rate, *burst)
	if *authToken != "" || *rate > 0 {
		log.Info("guard", "auth", *authToken != "", "rate", *rate, "burst", *burst)
	}

	srv := &http.Server{
		Addr:    *addr,
		Handler: guard.Wrap(api.New(scheduler, be, log).Routes()),
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

	// SIGTERM as well as interrupt. Interrupt alone is what a terminal sends;
	// SIGTERM is what every process supervisor sends -- systemd, Docker,
	// Kubernetes -- so without it a deployment kills generations mid-stream and
	// the graceful shutdown below never runs. It is one identifier and it is the
	// difference between "shuts down cleanly" and "shuts down cleanly when a
	// human does it by hand".
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	// The guard's bucket map is keyed by client address, so on a reachable
	// interface it grows with every address that has ever connected -- memory
	// exhaustion at one packet per key. A full bucket is indistinguishable from
	// one that does not exist, so the sweep is free to forget them.
	if *rate > 0 {
		go func() {
			tick := time.NewTicker(time.Minute)
			defer tick.Stop()
			for {
				select {
				case now := <-tick.C:
					if dropped := guard.Sweep(now, 10*time.Minute); dropped > 0 {
						log.Debug("forgot idle rate-limit buckets", "count", dropped)
					}
				case <-ctx.Done():
					return
				}
			}
		}()
	}

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

// loopbackOnly reports whether an address will only accept connections from
// this machine.
//
// The empty host in ":8080" is the trap: it means every interface, which reads
// like a default and behaves like a decision. It is treated as reachable here,
// which is why the default -addr does not start without a token.
func loopbackOnly(addr string) bool {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		// Not a host:port at all. Whatever it is, do not assume it is safe.
		return false
	}
	if host == "" {
		return false
	}
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(strings.Trim(host, "[]"))
	return ip != nil && ip.IsLoopback()
}
