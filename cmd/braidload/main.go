// Command braidload measures a running braid server, and prints the table.
//
// It sweeps a list of concurrency levels, and at each one holds that many
// clients in flight until it has collected the requested number of completed
// generations. What it reports per level is the shape of the whole argument for
// batching: how many forward passes the server needed, how many tokens came out
// of them, and what the tail of the time-to-first-token looked like while that
// happened.
//
// The numbers it prints are about whatever backend the server is running. With
// the mock that is a synthetic cost curve and the output is a test of this
// harness, not a claim about a model; the header says which backend answered so
// that a table pasted somewhere else still says it too.
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

type sample struct {
	ttft  time.Duration
	total time.Duration
	toks  int
	err   error
}

type stats struct {
	Scheduler struct {
		Steps     int64   `json:"steps"`
		Advanced  int64   `json:"sequences_advanced"`
		Tokens    int64   `json:"tokens"`
		Rejected  int64   `json:"rejected_queue_full"`
		Completed int64   `json:"completed"`
		Failed    int64   `json:"failed"`
		MeanBatch float64 `json:"mean_batch"`
	} `json:"scheduler"`

	// Step is absent when the backend cannot say where its time went, which is
	// the case for the mock. The columns that need it are then left blank
	// rather than filled with a zero that would read as a measurement.
	Step *struct {
		Steps     int64   `json:"steps"`
		WallMS    float64 `json:"wall_ms_per_step"`
		BuildMS   float64 `json:"build_ms_per_step"`
		ForwardMS float64 `json:"forward_ms_per_step"`
		CopyMS    float64 `json:"copy_ms_per_step"`
		SampleMS  float64 `json:"sample_ms_per_step"`
		PipeMS    float64 `json:"pipe_ms_per_step"`
		Kernels   float64 `json:"kernels_per_step"`

		// Cumulative totals, so a level's own figures come from the difference
		// between two snapshots rather than from the server's running mean.
		WallTotal    float64 `json:"-"`
		ForwardTotal float64 `json:"-"`
		PipeTotal    float64 `json:"-"`
	} `json:"step"`

	// Pool is present only when the server is running more than one worker.
	Pool *struct {
		Workers int `json:"workers"`
		Live    int `json:"live"`
	} `json:"pool"`
}

// backend names what answered, from what the server is able to report about
// itself: only a real worker can break a step down, and only a pool can lose
// one. Printed above the table so that a table pasted somewhere else still says
// what produced it.
func (s *stats) backend() string {
	switch {
	case s.Pool != nil:
		return fmt.Sprintf("pool of %d workers", s.Pool.Workers)
	case s.Step != nil:
		return "one worker"
	default:
		return "the MOCK backend: these numbers are about the scheduler, not a model"
	}
}

// totals turns the per-step means the server reports back into sums, so two
// snapshots can be subtracted to get one level in isolation.
func (s *stats) totals() (wall, forward, copyBack, sample, kernels float64, steps int64) {
	if s.Step == nil {
		return 0, 0, 0, 0, 0, 0
	}
	n := float64(s.Step.Steps)
	return s.Step.WallMS * n, s.Step.ForwardMS * n, s.Step.CopyMS * n, s.Step.SampleMS * n,
		s.Step.Kernels * n, s.Step.Steps
}

func main() {
	var (
		addr        = flag.String("addr", "http://127.0.0.1:8080", "server to measure")
		levels      = flag.String("concurrency", "1,2,4,8,16,32,64", "concurrency levels to sweep")
		perLevel    = flag.Int("requests", 64, "completed generations to collect at each level")
		maxTokens   = flag.Int("max-tokens", 100, "tokens per generation")
		temperature = flag.Float64("temperature", 0.8, "sampling temperature")
		svgDir      = flag.String("svg", "", "directory to write the README's charts into")
		sloFlag     = flag.Float64("slo-ms", 100,
			"time to first token a request must beat to count towards goodput")
		repeat = flag.Int("repeat", 1,
			"how many times to measure each level; above one the table reports a median and a range")
	)
	flag.Parse()

	client := &http.Client{Timeout: 10 * time.Minute}

	// Asked once, before anything is measured: a table is worth less to whoever
	// reads it later if it does not say what produced it.
	first, err := readStats(client, *addr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "reading stats from %s: %v\n", *addr, err)
		os.Exit(1)
	}
	fmt.Printf("Answered by %s.\n\n", first.backend())

	var charted []level

	// The last columns decompose one step: forward is the model's kernels, copy
	// is pulling the result off the device, sample is the softmax and the draw,
	// and pipe is what is left of the wall clock once those are taken off --
	// two writes, two reads and the serialising at each end.
	fmt.Printf("| clients | completed | steps | mean batch | tokens/s | TTFT p50 | TTFT p95 " +
		"| wall ms | forward ms | copy ms | sample ms | pipe ms | kernels |\n")
	fmt.Printf("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")

	for _, field := range strings.Split(*levels, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(field))
		if err != nil || n < 1 {
			fmt.Fprintf(os.Stderr, "not a concurrency level: %q\n", field)
			os.Exit(1)
		}
		// With fewer requests than clients, the extra clients never get one and
		// the level silently measures a smaller number than the one it prints.
		// A row labelled 64 that was really 48 is worse than no row at all.
		if *perLevel < n {
			fmt.Fprintf(os.Stderr,
				"concurrency %d needs at least that many requests to reach it; -requests is %d\n",
				n, *perLevel)
			os.Exit(1)
		}

		// Repetitions, because one reading of a timing has no error bar and this
		// machine is demonstrably noisy: the same sweep run in the other
		// direction moves the busiest row by a quarter. A median with the range
		// beside it says what a single number cannot.
		reps := make([]level, 0, *repeat)
		for range *repeat {
			before, err := readStats(client, *addr)
			if err != nil {
				fmt.Fprintf(os.Stderr, "reading stats: %v\n", err)
				os.Exit(1)
			}

			start := time.Now()
			samples := sweep(client, *addr, n, *perLevel, *maxTokens, float32(*temperature))
			elapsed := time.Since(start)

			after, err := readStats(client, *addr)
			if err != nil {
				fmt.Fprintf(os.Stderr, "reading stats: %v\n", err)
				os.Exit(1)
			}
			reps = append(reps, summarise(samples, before, after, elapsed, *sloFlag))
		}
		report(n, reps)

		// The median repetition, for the charts. Picking the median rather than
		// averaging keeps every point on a curve from a run that actually
		// happened, so a knee cannot be an artefact of two runs blended.
		if *svgDir != "" {
			chosen := medianRep(reps)
			chosen.clients = n
			charted = append(charted, chosen)
		}
	}

	if *svgDir != "" {
		if err := writeCharts(*svgDir, charted, *sloFlag); err != nil {
			fmt.Fprintf(os.Stderr, "writing charts: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("\ncharts written to %s\n", *svgDir)
	}
}

// medianRep is the repetition whose throughput is the median of the set.
func medianRep(reps []level) level {
	if len(reps) == 1 {
		return reps[0]
	}
	order := append([]level(nil), reps...)
	sort.Slice(order, func(i, j int) bool { return order[i].rate < order[j].rate })
	return order[len(order)/2]
}

// level is one measurement of one concurrency level: the scalars a row is made
// of, kept per repetition so the table can show a median and a spread rather
// than whichever run happened to be last.
type level struct {
	completed int
	rejected  int64
	steps     int64
	meanBatch float64
	rate      float64
	clients   int
	ttft50    float64
	ttft95    float64
	// p99 and goodput exist for the charts rather than the table. A p95 hides
	// the shape of an overloaded server, and a throughput figure that counts
	// tokens nobody waited for is not the throughput anybody is buying.
	ttft99  float64
	goodput float64
	// The raw times behind the percentiles, kept only when charts are asked
	// for: the distribution is the thing a percentile is a summary of.
	ttfts    []time.Duration
	wall     float64
	forward  float64
	copyBack float64
	sample   float64
	pipe     float64
	kernels  float64
	hasStep  bool
}

// sweep keeps n requests in flight until count of them have finished.
func sweep(client *http.Client, addr string, n, count, maxTokens int, temp float32) []sample {
	work := make(chan int, count)
	for i := range count {
		work <- i
	}
	close(work)

	out := make([]sample, 0, count)
	var mu sync.Mutex
	var wg sync.WaitGroup

	for range n {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for seed := range work {
				s := generate(client, addr, maxTokens, temp, uint64(seed))
				mu.Lock()
				out = append(out, s)
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	return out
}

// generate runs one request and times it, reading the stream token by token so
// that the first-token measurement is the client's, not the server's.
func generate(client *http.Client, addr string, maxTokens int, temp float32, seed uint64) sample {
	body, _ := json.Marshal(map[string]any{
		"prompt":      "the engine ",
		"max_tokens":  maxTokens,
		"temperature": temp,
		"seed":        seed,
	})

	start := time.Now()
	resp, err := client.Post(addr+"/v1/generate", "application/json", bytes.NewReader(body))
	if err != nil {
		return sample{err: err}
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return sample{err: fmt.Errorf("http %d", resp.StatusCode)}
	}

	var s sample
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		var payload struct {
			T     string `json:"t"`
			Error string `json:"error"`
		}
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &payload); err != nil {
			continue
		}
		if payload.Error != "" {
			s.err = fmt.Errorf("%s", payload.Error)
			break
		}
		if payload.T == "" {
			continue // the done event, which carries no token
		}
		if s.toks == 0 {
			s.ttft = time.Since(start)
		}
		s.toks++
	}
	if err := scanner.Err(); err != nil && s.err == nil {
		s.err = err
	}
	s.total = time.Since(start)
	return s
}

func summarise(samples []sample, before, after stats, elapsed time.Duration, sloMS float64) level {
	ttfts := make([]time.Duration, 0, len(samples))
	var out level
	var tokens int
	for _, s := range samples {
		if s.err != nil || s.toks == 0 {
			continue
		}
		out.completed++
		tokens += s.toks
		ttfts = append(ttfts, s.ttft)
	}

	out.steps = after.Scheduler.Steps - before.Scheduler.Steps
	if out.steps > 0 {
		out.meanBatch = float64(after.Scheduler.Advanced-before.Scheduler.Advanced) /
			float64(out.steps)
	}
	out.rate = float64(tokens) / elapsed.Seconds()
	out.rejected = after.Scheduler.Rejected - before.Scheduler.Rejected
	out.ttft50 = pctMS(ttfts, 50)
	out.ttft95 = pctMS(ttfts, 95)
	out.ttft99 = pctMS(ttfts, 99)

	// Goodput: the tokens that arrived inside the deadline, over the whole
	// wall clock. A server past its knee keeps producing tokens at a fine rate
	// while serving nobody in time, and the plain figure cannot tell the two
	// apart.
	var inTime int
	for _, s := range samples {
		if s.err != nil || s.toks == 0 {
			continue
		}
		if float64(s.ttft.Microseconds())/1000 <= sloMS {
			inTime += s.toks
		}
	}
	out.goodput = float64(inTime) / elapsed.Seconds()
	out.ttfts = ttfts

	wallA, fwdA, copyA, sampA, kernA, stepsA := after.totals()
	wallB, fwdB, copyB, sampB, kernB, stepsB := before.totals()
	if n := stepsA - stepsB; n > 0 {
		d := float64(n)
		out.hasStep = true
		out.wall = (wallA - wallB) / d
		out.forward = (fwdA - fwdB) / d
		out.copyBack = (copyA - copyB) / d
		out.sample = (sampA - sampB) / d
		out.pipe = ((wallA - fwdA - copyA - sampA) - (wallB - fwdB - copyB - sampB)) / d
		out.kernels = (kernA - kernB) / d
	}
	return out
}

// report prints one row out of however many repetitions there were: the median
// of each column, and for throughput the range as well, because that is the
// column this machine is worst at holding still.
func report(clients int, reps []level) {
	if len(reps) == 0 {
		return
	}
	med := func(pick func(level) float64) float64 {
		xs := make([]float64, len(reps))
		for i, r := range reps {
			xs[i] = pick(r)
		}
		sort.Float64s(xs)
		return xs[len(xs)/2]
	}

	rateText := fmt.Sprintf("%.0f", med(func(r level) float64 { return r.rate }))
	if len(reps) > 1 {
		lo, hi := reps[0].rate, reps[0].rate
		for _, r := range reps[1:] {
			lo, hi = min(lo, r.rate), max(hi, r.rate)
		}
		rateText = fmt.Sprintf("%.0f (%.0f-%.0f)", med(func(r level) float64 { return r.rate }), lo, hi)
	}

	// A backend that cannot break a step down leaves those columns empty rather
	// than filled with a zero somebody would read as a measurement.
	cell := func(pick func(level) float64) string {
		if !reps[0].hasStep {
			return "-"
		}
		return fmt.Sprintf("%.2f", med(pick))
	}

	fmt.Printf("| %d | %d | %.0f | %.2f | %s | %.0f ms | %.0f ms | %s | %s | %s | %s | %s | %s |\n",
		clients, reps[0].completed,
		med(func(r level) float64 { return float64(r.steps) }),
		med(func(r level) float64 { return r.meanBatch }),
		rateText,
		med(func(r level) float64 { return r.ttft50 }),
		med(func(r level) float64 { return r.ttft95 }),
		cell(func(r level) float64 { return r.wall }),
		cell(func(r level) float64 { return r.forward }),
		cell(func(r level) float64 { return r.copyBack }),
		cell(func(r level) float64 { return r.sample }),
		cell(func(r level) float64 { return r.pipe }),
		cell(func(r level) float64 { return r.kernels }))

	var rejected int64
	for _, r := range reps {
		rejected += r.rejected
	}
	if rejected > 0 {
		fmt.Fprintf(os.Stderr, "  %d requests rejected at %d clients: the queue filled\n",
			rejected, clients)
	}
}

// pctMS is the nearest-rank percentile in milliseconds: the smallest
// observation at or above the given share of the sorted sample. No
// interpolation, because interpolating between two measurements invents a third
// that nobody observed.
func pctMS(ds []time.Duration, p int) float64 {
	if len(ds) == 0 {
		return 0
	}
	sorted := append([]time.Duration(nil), ds...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })

	rank := (p*len(sorted) + 99) / 100 // ceil(p/100 * n)
	if rank < 1 {
		rank = 1
	}
	return float64(sorted[rank-1].Microseconds()) / 1000
}

func readStats(client *http.Client, addr string) (stats, error) {
	resp, err := client.Get(addr + "/stats")
	if err != nil {
		return stats{}, err
	}
	defer resp.Body.Close()

	var s stats
	err = json.NewDecoder(resp.Body).Decode(&s)
	return s, err
}
