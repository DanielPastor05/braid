package api

import (
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/DanielPastor05/braid/internal/backend"
	"github.com/DanielPastor05/braid/internal/sched"
)

// parseExposition is a small reader for the Prometheus text format, strict
// enough to catch the ways this file could be wrong.
//
// Checking for a 200 and a substring would pass on output no scraper accepts: a
// TYPE line for a name that never appears, a sample whose value is not a number,
// a metric declared twice. Those are the mistakes a hand-written exporter makes,
// so the test has to be a parser rather than a search.
func parseExposition(t *testing.T, body string) map[string]float64 {
	t.Helper()

	samples := map[string]float64{}
	declaredType := map[string]string{}
	declaredHelp := map[string]bool{}

	for i, line := range strings.Split(strings.TrimSpace(body), "\n") {
		line = strings.TrimRight(line, "\r")
		if line == "" {
			continue
		}
		where := "line " + strconv.Itoa(i+1)

		if after, found := strings.CutPrefix(line, "# HELP "); found {
			name, help, ok := strings.Cut(after, " ")
			if !ok || help == "" {
				t.Errorf("%s: HELP for %q has no text", where, name)
			}
			if declaredHelp[name] {
				t.Errorf("%s: %q declared HELP twice", where, name)
			}
			declaredHelp[name] = true
			continue
		}
		if after, found := strings.CutPrefix(line, "# TYPE "); found {
			name, kind, ok := strings.Cut(after, " ")
			if !ok {
				t.Errorf("%s: TYPE without a kind", where)
				continue
			}
			switch kind {
			case "counter", "gauge", "histogram", "summary", "untyped":
			default:
				t.Errorf("%s: %q is not a Prometheus metric type", where, kind)
			}
			declaredType[name] = kind
			continue
		}
		if strings.HasPrefix(line, "#") {
			continue
		}

		name, value, ok := strings.Cut(line, " ")
		if !ok {
			t.Errorf("%s: %q is not `name value`", where, line)
			continue
		}
		got, err := strconv.ParseFloat(value, 64)
		if err != nil {
			t.Errorf("%s: %q is not a number for %s", where, value, name)
			continue
		}
		if _, seen := samples[name]; seen {
			t.Errorf("%s: %s appears twice", where, name)
		}
		samples[name] = got
	}

	// Every sample must have been declared, and every declaration must have a
	// sample. A TYPE line for a name that never appears is the classic way a
	// hand-written exporter drifts.
	for name := range samples {
		if declaredType[name] == "" {
			t.Errorf("%s has a sample but no TYPE", name)
		}
		if !declaredHelp[name] {
			t.Errorf("%s has a sample but no HELP", name)
		}
	}
	for name := range declaredType {
		if _, ok := samples[name]; !ok {
			t.Errorf("%s was declared but never sampled", name)
		}
	}
	return samples
}

func TestMetricsAreValidExposition(t *testing.T) {
	srv := serve(t, sched.Config{MaxBatch: 4, QueueDepth: 8, MaxTokensLimit: 64}, nil)

	// Generate something, so the counters are not all zero and the step timings
	// have a step to report.
	resp := post(t, srv, `{"prompt":"hello","max_tokens":5,"temperature":0.7,"seed":1}`)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("the warm-up generation got %d", resp.StatusCode)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	resp.Body.Close()

	body, contentType := get(t, srv, "/metrics")
	if !strings.HasPrefix(contentType, "text/plain") {
		t.Errorf("Content-Type is %q; a scraper wants text/plain", contentType)
	}

	samples := parseExposition(t, body)

	// The counters that have to have moved, because a request just went through.
	for _, name := range []string{
		"braid_requests_accepted_total",
		"braid_requests_completed_total",
		"braid_steps_total",
		"braid_tokens_total",
		"braid_mean_batch",
		"braid_mean_width",
	} {
		value, ok := samples[name]
		if !ok {
			t.Errorf("%s is missing", name)
			continue
		}
		if value <= 0 {
			t.Errorf("%s is %g after a completed generation", name, value)
		}
	}

	if samples["braid_tokens_total"] != 5 {
		t.Errorf("braid_tokens_total is %g after asking for 5 tokens", samples["braid_tokens_total"])
	}
	// Every counter that should exist even at zero. A counter that is absent
	// until it is non-zero makes rate() wrong across a restart.
	for _, name := range []string{
		"braid_requests_rejected_total",
		"braid_requests_failed_total",
		"braid_step_errors_total",
	} {
		if _, ok := samples[name]; !ok {
			t.Errorf("%s is missing at zero", name)
		}
	}
}

// TestMetricsOnAnIdleServerAreStillValid: the exporter runs before anything has
// happened too, and a division by a zero step count is the obvious way to emit
// NaN -- which parses as a float and poisons a dashboard silently.
func TestMetricsOnAnIdleServerAreStillValid(t *testing.T) {
	srv := serve(t, sched.Config{MaxBatch: 4, QueueDepth: 8, MaxTokensLimit: 64}, nil)

	body, _ := get(t, srv, "/metrics")
	samples := parseExposition(t, body)

	for name, value := range samples {
		if value != value { // NaN
			t.Errorf("%s is NaN on an idle server", name)
		}
	}
	if len(samples) == 0 {
		t.Error("an idle server exported nothing at all")
	}
}

// TestMetricsAreBehindTheGuard: an unauthenticated /metrics on a reachable
// interface hands the shape of the workload to anybody who asks.
func TestMetricsAreBehindTheGuard(t *testing.T) {
	mock := backend.NewMock()
	mock.Base, mock.PerSeq = 0, 0
	s, err := sched.New(mock, sched.Config{MaxBatch: 4, QueueDepth: 8, MaxTokensLimit: 64})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = s.Close() })

	h := NewGuard("s3cret", 0, 0).Wrap(
		New(s, mock, slog.New(slog.NewTextHandler(io.Discard, nil))).Routes())

	ask := func(token string) int {
		r := httptest.NewRequest(http.MethodGet, "/metrics", nil)
		r.RemoteAddr = "10.0.0.1:1234"
		if token != "" {
			r.Header.Set("Authorization", "Bearer "+token)
		}
		w := httptest.NewRecorder()
		h.ServeHTTP(w, r)
		return w.Code
	}

	if code := ask(""); code != http.StatusUnauthorized {
		t.Errorf("/metrics answered %d without a token", code)
	}
	if code := ask("s3cret"); code != http.StatusOK {
		t.Errorf("/metrics answered %d with the right token", code)
	}
}

// get reads a body and its content type, and fails the test rather than the
// caller having to.
func get(t *testing.T, srv *httptest.Server, path string) (string, string) {
	t.Helper()
	resp, err := srv.Client().Get(srv.URL + path)
	if err != nil {
		t.Fatalf("GET %s: %v", path, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET %s got %d", path, resp.StatusCode)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}
	return string(body), resp.Header.Get("Content-Type")
}
