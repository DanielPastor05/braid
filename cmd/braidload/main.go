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
	Steps     int64   `json:"steps"`
	Advanced  int64   `json:"sequences_advanced"`
	Tokens    int64   `json:"tokens"`
	Rejected  int64   `json:"rejected_queue_full"`
	Completed int64   `json:"completed"`
	Failed    int64   `json:"failed"`
	MeanBatch float64 `json:"mean_batch"`
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

	fmt.Printf("| clients | completed | rejected | steps | mean batch | tokens/s | TTFT p50 | TTFT p95 | TTFT p99 | total p50 |\n")
	fmt.Printf("|---|---|---|---|---|---|---|---|---|---|\n")

	for _, field := range strings.Split(*levels, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(field))
		if err != nil || n < 1 {
			fmt.Fprintf(os.Stderr, "not a concurrency level: %q\n", field)
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
	totals := make([]time.Duration, 0, len(samples))
	var okCount, tokens int
	for _, s := range samples {
		if s.err != nil || s.toks == 0 {
			continue
		}
		okCount++
		tokens += s.toks
		ttfts = append(ttfts, s.ttft)
		totals = append(totals, s.total)
	}

	steps := after.Steps - before.Steps
	advanced := after.Advanced - before.Advanced
	meanBatch := 0.0
	if steps > 0 {
		meanBatch = float64(advanced) / float64(steps)
	}
	rate := float64(tokens) / elapsed.Seconds()

	fmt.Printf("| %d | %d | %d | %d | %.2f | %.0f | %s | %s | %s | %s |\n",
		clients, okCount, after.Rejected-before.Rejected, steps, meanBatch, rate,
		pct(ttfts, 50), pct(ttfts, 95), pct(ttfts, 99), pct(totals, 50))
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
