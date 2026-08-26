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
func (s *stats) totals() (wall, forward, build, sample, kernels float64, steps int64) {
	if s.Step == nil {
		return 0, 0, 0, 0, 0, 0
	}
	n := float64(s.Step.Steps)
	return s.Step.WallMS * n, s.Step.ForwardMS * n, s.Step.BuildMS * n, s.Step.SampleMS * n,
		s.Step.Kernels * n, s.Step.Steps
}

func main() {
	var (
		addr        = flag.String("addr", "http://127.0.0.1:8080", "server to measure")
		levels      = flag.String("concurrency", "1,2,4,8,16,32,64", "concurrency levels to sweep")
		perLevel    = flag.Int("requests", 64, "completed generations to collect at each level")
		maxTokens   = flag.Int("max-tokens", 100, "tokens per generation")
		temperature = flag.Float64("temperature", 0.8, "sampling temperature")
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

	// The last four columns decompose one step: build fills the (n, 64) tensor,
	// forward is the model including the copy off the device, sample is the
	// softmax and the draw, and pipe is what is left of the wall clock once
	// those are taken off -- two writes, two reads and the serialising.
	fmt.Printf("| clients | completed | steps | mean batch | tokens/s | TTFT p50 | TTFT p95 | wall ms | build ms | forward ms | sample ms | pipe ms |\n")
	fmt.Printf("|---|---|---|---|---|---|---|---|---|---|---|---|\n")

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

		report(n, samples, before, after, elapsed)
	}
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

func report(clients int, samples []sample, before, after stats, elapsed time.Duration) {
	ttfts := make([]time.Duration, 0, len(samples))
	var okCount, tokens int
	for _, s := range samples {
		if s.err != nil || s.toks == 0 {
			continue
		}
		okCount++
		tokens += s.toks
		ttfts = append(ttfts, s.ttft)
	}

	steps := after.Scheduler.Steps - before.Scheduler.Steps
	advanced := after.Scheduler.Advanced - before.Scheduler.Advanced
	meanBatch := 0.0
	if steps > 0 {
		meanBatch = float64(advanced) / float64(steps)
	}
	rate := float64(tokens) / elapsed.Seconds()

	// The split for this level alone: the difference between two cumulative
	// snapshots, divided by the steps between them. Reading the server's own
	// running mean instead would fold every earlier level into every later one.
	wall, forward, build, sample := "-", "-", "-", "-"
	pipe, kernels := "-", "-"
	wallA, fwdA, buildA, sampA, kernA, stepsA := after.totals()
	wallB, fwdB, buildB, sampB, kernB, stepsB := before.totals()
	if n := stepsA - stepsB; n > 0 {
		each := func(a, b float64) string {
			return fmt.Sprintf("%.2f", (a-b)/float64(n))
		}
		wall = each(wallA, wallB)
		forward = each(fwdA, fwdB)
		build = each(buildA, buildB)
		sample = each(sampA, sampB)
		pipe = each((wallA - fwdA - buildA - sampA), (wallB - fwdB - buildB - sampB))
		kernels = fmt.Sprintf("%.0f", (kernA-kernB)/float64(n))
	}

	fmt.Printf("| %d | %d | %d | %.2f | %.0f | %s | %s | %s | %s | %s | %s | %s | %s |\n",
		clients, okCount, steps, meanBatch, rate,
		pct(ttfts, 50), pct(ttfts, 95), wall, build, forward, sample, pipe, kernels)

	// Rejections do not get a column of zeros. They get a line, on stderr, on
	// the runs where they actually happened.
	if r := after.Scheduler.Rejected - before.Scheduler.Rejected; r > 0 {
		fmt.Fprintf(os.Stderr, "  %d requests rejected at %d clients: the queue filled\n", r, clients)
	}
}

// pct is the nearest-rank percentile: the smallest observation at or above the
// given share of the sorted sample. No interpolation, because interpolating
// between two measurements invents a third that nobody observed.
func pct(ds []time.Duration, p int) string {
	if len(ds) == 0 {
		return "-"
	}
	sorted := append([]time.Duration(nil), ds...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })

	rank := (p*len(sorted) + 99) / 100 // ceil(p/100 * n)
	if rank < 1 {
		rank = 1
	}
	d := sorted[rank-1]
	return fmt.Sprintf("%.0f ms", float64(d.Microseconds())/1000)
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
