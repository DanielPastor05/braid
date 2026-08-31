// Package api puts an HTTP surface on the scheduler.
//
// Tokens are streamed as server-sent events, which is the transport that
// matches what the scheduler produces: one small message at a time, in order,
// over a connection the client may drop at any moment. A dropped connection
// cancels the request context, the scheduler notices at its next step, and the
// sequence leaves the batch instead of generating into a socket nobody is
// reading.
package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"github.com/DanielPastor05/braid/internal/backend"
	"github.com/DanielPastor05/braid/internal/sched"
)

// Server is the HTTP front of one scheduler.
type Server struct {
	sched   *sched.Scheduler
	backend backend.Backend
	log     *slog.Logger
}

func New(s *sched.Scheduler, be backend.Backend, log *slog.Logger) *Server {
	return &Server{sched: s, backend: be, log: log}
}

// Routes returns the mux. Kept separate from any listener so tests can drive it
// through httptest without opening a port.
func (s *Server) Routes() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/generate", s.generate)
	mux.HandleFunc("GET /healthz", s.healthz)
	mux.HandleFunc("GET /stats", s.stats)
	mux.HandleFunc("GET /metrics", s.metrics)
	return mux
}

type generateRequest struct {
	Prompt      string  `json:"prompt"`
	MaxTokens   int     `json:"max_tokens"`
	Temperature float32 `json:"temperature"`
	Seed        uint64  `json:"seed"`

	// MaxWaitMS bounds queue time. Zero means the request waits as long as it
	// takes; a client with a deadline of its own should say so here, because a
	// request rejected at admission costs the server nothing and a request
	// abandoned after the GPU has worked on it costs it everything.
	MaxWaitMS int `json:"max_wait_ms"`
}

type doneEvent struct {
	Generated  int     `json:"generated"`
	QueuedMS   float64 `json:"queued_ms"`
	FirstTokMS float64 `json:"first_token_ms"`
	TotalMS    float64 `json:"total_ms"`
}

func (s *Server) generate(w http.ResponseWriter, r *http.Request) {
	var req generateRequest
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20)).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, fmt.Sprintf("malformed request body: %v", err))
		return
	}
	if req.Temperature == 0 {
		req.Temperature = 0.8
	}
	if req.MaxTokens == 0 {
		req.MaxTokens = 200
	}

	tokens, done, err := s.sched.Submit(r.Context(), sched.Request{
		Prompt:      req.Prompt,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		Seed:        req.Seed,
		MaxWait:     time.Duration(req.MaxWaitMS) * time.Millisecond,
	})
	switch {
	case errors.Is(err, sched.ErrQueueFull):
		// 429 rather than a queue that grows without bound. The server would
		// rather say no now than say yes and answer in a minute.
		w.Header().Set("Retry-After", "1")
		writeError(w, http.StatusTooManyRequests, "the queue is full")
		return
	case errors.Is(err, sched.ErrClosed):
		writeError(w, http.StatusServiceUnavailable, "the server is shutting down")
		return
	case err != nil:
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Accel-Buffering", "no") // proxies must not hold the stream
	w.WriteHeader(http.StatusOK)

	rc := http.NewResponseController(w)
	_ = rc.Flush() // headers reach the client before the first token

	for tok := range tokens {
		if err := writeEvent(w, "token", map[string]string{"t": tok.Text}); err != nil {
			// The client is gone. Draining the channel rather than returning
			// here would keep the sequence in the batch; instead the request
			// context is already cancelled by net/http, and the scheduler drops
			// the sequence at its next step.
			s.log.Debug("client went away mid-stream", "error", err)
			return
		}
		_ = rc.Flush()
	}

	res := <-done
	if res.Err != nil {
		// The status line is long gone by now, so a failure has to arrive as an
		// event. A client that ignores this will see a short answer and no
		// explanation, which is why the event is named and not just a comment.
		_ = writeEvent(w, "error", map[string]string{"error": res.Err.Error()})
		_ = rc.Flush()
		return
	}

	_ = writeEvent(w, "done", doneEvent{
		Generated:  res.Generated,
		QueuedMS:   ms(res.Queued),
		FirstTokMS: ms(res.FirstTok),
		TotalMS:    ms(res.Total),
	})
	_ = rc.Flush()
}

func (s *Server) healthz(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok"})
}

// timed is what a backend implements when it can say where a step's time went.
// It is an optional interface rather than part of Backend, because the mock has
// nothing honest to report and the scheduler does not care either way.
type timed interface {
	Timings() backend.Timings
}

// pooled is what a backend implements when it is more than one process and some
// of them may have died since the server started.
type pooled interface {
	PoolStats() backend.PoolStats
}

func (s *Server) stats(w http.ResponseWriter, _ *http.Request) {
	payload := map[string]any{"scheduler": s.sched.Stats()}
	if t, ok := s.backend.(timed); ok {
		payload["step"] = t.Timings()
	}
	if p, ok := s.backend.(pooled); ok {
		payload["pool"] = p.PoolStats()
	}
	writeJSON(w, http.StatusOK, payload)
}

func ms(d time.Duration) float64 {
	return float64(d.Microseconds()) / 1000
}

func writeEvent(w http.ResponseWriter, name string, payload any) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", name, body)
	return err
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, map[string]string{"error": message})
}
