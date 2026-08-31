package main

import (
	"fmt"
	"math"
	"math/rand/v2"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Open-loop arrivals, because the closed loop flatters the server and this page
// quotes its numbers.
//
// `sweep` holds n requests in flight and starts a new one whenever one finishes.
// That is a closed loop, and a closed loop cannot overload anything: the arrival
// rate is throttled by the completion rate, so as the server slows down the load
// politely slows down with it. Every tail latency measured that way is the tail
// of a system that was never pushed past what it could take.
//
// Real traffic does not wait. Requests arrive when they arrive, and the queue
// grows when they arrive faster than they leave -- which is the regime the
// interesting failures live in and the one the closed loop cannot reach.
//
// So: arrivals at a fixed mean rate with exponential gaps, which is a Poisson
// process. Exponential rather than a fixed interval because evenly spaced
// arrivals never coincide, and coincidence is what makes a queue.

// arrive fires requests at `rate` per second for `duration`, without waiting for
// any of them, and returns every sample.
//
// The number in flight is not bounded here on purpose. Bounding it would be the
// closed loop again by another name, and what this is for is the case where more
// arrive than can be served. What bounds it in practice is the server: it
// refuses with a 429 once its queue is full, and those refusals are part of the
// measurement rather than an error in it.
func arrive(client *http.Client, addr string, rate float64, duration time.Duration,
	maxTokens int, temp float32, seed uint64) []sample {

	rng := rand.New(rand.NewPCG(seed, 0x9E3779B97F4A7C15))

	var mu sync.Mutex
	var wg sync.WaitGroup
	out := make([]sample, 0, int(rate*duration.Seconds())+16)

	deadline := time.Now().Add(duration)
	for n := uint64(0); time.Now().Before(deadline); n++ {
		wg.Add(1)
		go func(s uint64) {
			defer wg.Done()
			result := generate(client, addr, maxTokens, temp, s)
			mu.Lock()
			out = append(out, result)
			mu.Unlock()
		}(seed + n)

		// The gap to the next arrival: exponential with mean 1/rate. Drawing it
		// from a uniform through -ln(u)/rate is the inverse-CDF method, the same
		// one the sampler in the worker uses on a different distribution.
		gap := time.Duration(-math.Log(1-rng.Float64()) / rate * float64(time.Second))
		if until := time.Until(deadline); gap > until {
			break
		}
		time.Sleep(gap)
	}

	// Everything already sent is still owed an answer. A run that stopped
	// measuring at the deadline would drop precisely the slowest requests, which
	// are the ones the open loop exists to see.
	wg.Wait()
	return out
}

// summariseOpen turns an open-loop run into a level, with the offered rate
// recorded because at overload it is the independent variable and the completed
// count is what depends on it.
func summariseOpen(samples []sample, before, after stats, elapsed time.Duration,
	offered float64, sloMS float64) level {

	out := summarise(samples, before, after, elapsed, sloMS)
	out.offered = offered

	// Refusals are a result, not a failure. A server that says no quickly is
	// behaving correctly under overload, and a run that counted a 429 as an
	// error would report the well-behaved server as the broken one.
	for _, s := range samples {
		if s.err != nil {
			out.refused++
		}
	}
	return out
}

// runOpenLoop sweeps offered arrival rates and prints what the server did with
// each, which is a different table from the closed loop's because the columns
// mean different things.
//
// The closed loop asks "with n clients, how fast?". This asks "offered r
// requests a second, how many arrived, how many were served, and what did the
// ones that were served wait?" -- and past capacity the answers stop tracking
// the question, which is the point.
func runOpenLoop(client *http.Client, addr, spec string, hold time.Duration,
	maxTokens int, temp float32, sloMS float64) {

	fmt.Printf("| offered /s | sent | completed | refused | tokens/s | goodput/s | " +
		"TTFT p50 | TTFT p95 | TTFT p99 | mean batch |\n")
	fmt.Printf("|---|---|---|---|---|---|---|---|---|---|\n")

	var seed uint64 = 20260831
	for _, field := range strings.Split(spec, ",") {
		rate, err := strconv.ParseFloat(strings.TrimSpace(field), 64)
		if err != nil || rate <= 0 {
			fmt.Fprintf(os.Stderr, "not an arrival rate: %q\n", field)
			os.Exit(1)
		}

		before, err := readStats(client, addr)
		if err != nil {
			fmt.Fprintf(os.Stderr, "reading stats: %v\n", err)
			os.Exit(1)
		}

		start := time.Now()
		samples := arrive(client, addr, rate, hold, maxTokens, temp, seed)
		elapsed := time.Since(start)
		seed += uint64(len(samples)) + 1

		after, err := readStats(client, addr)
		if err != nil {
			fmt.Fprintf(os.Stderr, "reading stats: %v\n", err)
			os.Exit(1)
		}

		l := summariseOpen(samples, before, after, elapsed, rate, sloMS)
		fmt.Printf("| %.0f | %d | %d | %d | %.0f | %.0f | %.0f ms | %.0f ms | %.0f ms | %.2f |\n",
			l.offered, len(samples), l.completed, l.refused,
			l.rate, l.goodput, l.ttft50, l.ttft95, l.ttft99, l.meanBatch)
	}
}
