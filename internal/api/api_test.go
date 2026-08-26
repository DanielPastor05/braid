package api

import (
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/DanielPastor05/braid/internal/backend"
	"github.com/DanielPastor05/braid/internal/sched"
)

// The HTTP surface, against the mock backend. No process, no GPU, no
// checkpoint: everything here is about the contract a client sees.
//
// It is worth testing separately from the scheduler because the failures are
// different in kind. A scheduler bug produces wrong text; an API bug produces
// text nobody can read, or a status code that tells a client to retry something
// it should not.

func serve(t *testing.T, cfg sched.Config, tune func(*backend.Mock)) *httptest.Server {
	t.Helper()

	mock := backend.NewMock()
	mock.Base = 0
	mock.PerSeq = 0
	if tune != nil {
		tune(mock)
	}

	s, err := sched.New(mock, cfg)
	if err != nil {
		t.Fatalf("scheduler: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })

	srv := httptest.NewServer(New(s, mock, slog.New(slog.NewTextHandler(io.Discard, nil))).Routes())
	t.Cleanup(srv.Close)
	return srv
}

func post(t *testing.T, srv *httptest.Server, body string) *http.Response {
	t.Helper()

	resp, err := srv.Client().Post(srv.URL+"/v1/generate", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	return resp
}

// TestGenerateStreamsTokensThenDone pins the wire format. A client parsing
// server-sent events needs the event names and the shape of each payload, and
// changing either is the kind of break that no compiler catches and every
// client notices at once.
func TestGenerateStreamsTokensThenDone(t *testing.T) {
	srv := serve(t, sched.Config{MaxBatch: 4, QueueDepth: 8, MaxTokensLimit: 64}, nil)

	resp := post(t, srv, `{"prompt":"hello","max_tokens":5,"temperature":0.8,"seed":1}`)
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("content type %q, wanted text/event-stream", ct)
	}
	// Proxies that buffer would hold every token until the stream ended, which
	// turns a streaming API into a slow non-streaming one.
	if resp.Header.Get("X-Accel-Buffering") != "no" {
		t.Error("the response does not tell proxies to leave the stream alone")
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("reading the stream: %v", err)
	}
	text := string(body)

	if got := strings.Count(text, "event: token"); got != 5 {
		t.Errorf("asked for 5 tokens, the stream carried %d", got)
	}
	if !strings.Contains(text, "event: done") {
		t.Error("the stream ended without a done event")
	}
	if strings.Contains(text, "event: error") {
		t.Errorf("an error event arrived on a healthy request:\n%s", text)
	}

	// Every data line has to be JSON, because a client parses it before it can
	// tell whether it was a token or the end.
	for _, line := range strings.Split(text, "\n") {
		payload, found := strings.CutPrefix(line, "data: ")
		if !found {
			continue
		}
		var any map[string]any
		if err := json.Unmarshal([]byte(payload), &any); err != nil {
			t.Errorf("a data line was not JSON: %q", payload)
		}
	}

	// The done event carries what the client needs to report its own latency.
	last := text[strings.LastIndex(text, "event: done"):]
	var done struct {
		Generated  int     `json:"generated"`
		FirstTokMS float64 `json:"first_token_ms"`
		TotalMS    float64 `json:"total_ms"`
	}
	payload := last[strings.Index(last, "data: ")+len("data: "):]
	if err := json.Unmarshal([]byte(strings.SplitN(payload, "\n", 2)[0]), &done); err != nil {
		t.Fatalf("the done event was not JSON: %v", err)
	}
	if done.Generated != 5 {
		t.Errorf("the done event says %d tokens, the stream carried 5", done.Generated)
	}
	if done.TotalMS < done.FirstTokMS {
		t.Errorf("total %.3f ms is less than time to first token %.3f ms",
			done.TotalMS, done.FirstTokMS)
	}
}

// TestAFullQueueIsAFourTwentyNine is the backpressure contract. It has to be a
// status code and not an error event, because the client needs to know before
// it starts reading a stream that will never come -- and it has to be 429 with
// a Retry-After rather than 500, because this is a server saying "not now", not
// a server saying "something broke".
func TestAFullQueueIsAFourTwentyNine(t *testing.T) {
	// One sequence at a time, a queue of one, and a backend slow enough that
	// the queue cannot drain while the requests pile up.
	srv := serve(t, sched.Config{MaxBatch: 1, QueueDepth: 1, MaxTokensLimit: 512},
		func(m *backend.Mock) { m.Base = 50 * time.Millisecond })

	var rejected, accepted int
	for range 12 {
		resp := post(t, srv, `{"prompt":"x","max_tokens":200,"temperature":0.8}`)
		switch resp.StatusCode {
		case http.StatusTooManyRequests:
			rejected++
			if resp.Header.Get("Retry-After") == "" {
				t.Error("a 429 arrived without a Retry-After")
			}
			var body map[string]string
			if err := json.NewDecoder(resp.Body).Decode(&body); err != nil || body["error"] == "" {
				t.Error("a 429 arrived without a JSON error body")
			}
		case http.StatusOK:
			accepted++
		default:
			t.Errorf("unexpected status %d", resp.StatusCode)
		}
		resp.Body.Close()
	}

	if rejected == 0 {
		t.Fatal("a queue of one accepted twelve requests without rejecting any")
	}
	if accepted == 0 {
		t.Fatal("every request was rejected, so this measured a broken server rather than backpressure")
	}
}

// TestNonsenseIsRejectedBeforeTheStream checks the other rejection path: a
// request the scheduler will not take at all still gets a status code, because
// nothing has been written yet.
func TestNonsenseIsRejectedBeforeTheStream(t *testing.T) {
	srv := serve(t, sched.Config{MaxBatch: 4, QueueDepth: 8, MaxTokensLimit: 32}, nil)

	for _, body := range []string{
		`{"prompt":"x","max_tokens":10,"temperature":-1}`, // temperature must be positive
		`{"prompt":"x","max_tokens":9999}`,                // over MaxTokensLimit
		`not json at all`,
	} {
		resp := post(t, srv, body)
		if resp.StatusCode != http.StatusBadRequest {
			t.Errorf("%s got status %d, wanted 400", body, resp.StatusCode)
		}
		resp.Body.Close()
	}
}

// TestStatsReportsTheSchedulerAndTheBackend covers the endpoint the load
// harness reads. The harness subtracts two snapshots to get one level's figures,
// so the shape matters as much as the numbers.
func TestStatsReportsTheSchedulerAndTheBackend(t *testing.T) {
	srv := serve(t, sched.Config{MaxBatch: 4, QueueDepth: 8, MaxTokensLimit: 64}, nil)

	resp := post(t, srv, `{"prompt":"x","max_tokens":3,"temperature":0.8}`)
	_, _ = io.Copy(io.Discard, resp.Body)
	resp.Body.Close()

	stats, err := srv.Client().Get(srv.URL + "/stats")
	if err != nil {
		t.Fatalf("stats: %v", err)
	}
	defer stats.Body.Close()

	var payload map[string]json.RawMessage
	if err := json.NewDecoder(stats.Body).Decode(&payload); err != nil {
		t.Fatalf("stats was not JSON: %v", err)
	}
	if _, ok := payload["scheduler"]; !ok {
		t.Fatal("stats has no scheduler section")
	}
	// The mock cannot say where a step's time went, so the section the harness
	// treats as optional has to actually be absent rather than present and zero.
	if _, ok := payload["step"]; ok {
		t.Error("the mock backend reported a step breakdown it cannot know")
	}

	var scheduler struct {
		Steps     int64   `json:"steps"`
		Completed int64   `json:"completed"`
		MeanBatch float64 `json:"mean_batch"`
	}
	if err := json.Unmarshal(payload["scheduler"], &scheduler); err != nil {
		t.Fatalf("the scheduler section was not the shape the harness reads: %v", err)
	}
	if scheduler.Completed != 1 || scheduler.Steps != 3 {
		t.Errorf("after one three-token request: %d completed, %d steps",
			scheduler.Completed, scheduler.Steps)
	}
}

func TestHealthzAnswers(t *testing.T) {
	srv := serve(t, sched.Config{MaxBatch: 1, QueueDepth: 1, MaxTokensLimit: 8}, nil)

	resp, err := srv.Client().Get(srv.URL + "/healthz")
	if err != nil {
		t.Fatalf("healthz: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("status %d", resp.StatusCode)
	}
}
