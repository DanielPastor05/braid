package api

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func okHandler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
}

func request(t *testing.T, h http.Handler, token, from string) int {
	t.Helper()
	r := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	r.RemoteAddr = from
	if token != "" {
		r.Header.Set("Authorization", "Bearer "+token)
	}
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	return w.Code
}

// TestAGuardWithNothingSetAllowsEverything is the loopback case, and it has to
// stay cheap: the server binds 127.0.0.1 by default and a local run should not
// need a token to say hello.
func TestAGuardWithNothingSetAllowsEverything(t *testing.T) {
	h := NewGuard("", 0, 0).Wrap(okHandler())
	for range 100 {
		if code := request(t, h, "", "10.0.0.1:1234"); code != http.StatusOK {
			t.Fatalf("an unconfigured guard rejected a request with %d", code)
		}
	}
}

func TestTheTokenIsRequiredAndChecked(t *testing.T) {
	h := NewGuard("s3cret", 0, 0).Wrap(okHandler())

	cases := []struct {
		name  string
		token string
		want  int
	}{
		{"the right token", "s3cret", http.StatusOK},
		{"no token", "", http.StatusUnauthorized},
		{"the wrong token", "s3cre", http.StatusUnauthorized},
		{"a longer wrong token", "s3crets", http.StatusUnauthorized},
	}
	for _, c := range cases {
		if code := request(t, h, c.token, "10.0.0.1:1234"); code != c.want {
			t.Errorf("%s: got %d, want %d", c.name, code, c.want)
		}
	}
}

// TestAMalformedAuthorizationHeaderIsRejected covers the shapes that are not
// "Bearer <token>" at all. A prefix check that accepted them would be a hole
// with no error message.
func TestAMalformedAuthorizationHeaderIsRejected(t *testing.T) {
	g := NewGuard("s3cret", 0, 0)
	h := g.Wrap(okHandler())

	for _, header := range []string{"s3cret", "Basic s3cret", "Bearer", "bearer s3cret", ""} {
		r := httptest.NewRequest(http.MethodGet, "/healthz", nil)
		r.RemoteAddr = "10.0.0.1:1234"
		if header != "" {
			r.Header.Set("Authorization", header)
		}
		w := httptest.NewRecorder()
		h.ServeHTTP(w, r)
		if w.Code != http.StatusUnauthorized {
			t.Errorf("Authorization %q was accepted with %d", header, w.Code)
		}
	}
}

// TestTheBucketRefillsRatherThanResetting is the property that makes a bucket
// worth more than a counter in a window.
//
// A fixed window lets a client spend its whole allowance in the last instant of
// one window and again in the first of the next -- twice the burst, at the seam.
// A bucket has no seam: it is empty until time has actually passed.
func TestTheBucketRefillsRatherThanResetting(t *testing.T) {
	g := NewGuard("", 10, 3) // 10/s, burst 3
	start := time.Now()

	for i := range 3 {
		if !g.allow("client", start) {
			t.Fatalf("request %d of the burst was rejected", i)
		}
	}
	if g.allow("client", start) {
		t.Fatal("the fourth request went through: the burst is not a burst")
	}

	// A tenth of a second is one token at 10/s. Not two.
	if !g.allow("client", start.Add(100*time.Millisecond)) {
		t.Error("no token had appeared after a tenth of a second")
	}
	if g.allow("client", start.Add(100*time.Millisecond)) {
		t.Error("two tokens appeared where the rate allows one")
	}

	// And it never fills past the burst, however long nobody calls.
	for i := range 3 {
		if !g.allow("client", start.Add(time.Hour)) {
			t.Fatalf("request %d after an idle hour was rejected", i)
		}
	}
	if g.allow("client", start.Add(time.Hour)) {
		t.Error("an idle hour bought more than the burst")
	}
}

func TestClientsAreLimitedSeparately(t *testing.T) {
	g := NewGuard("", 1, 1)
	now := time.Now()

	if !g.allow("a", now) || !g.allow("b", now) {
		t.Fatal("two different clients could not each make a first request")
	}
	if g.allow("a", now) {
		t.Error("client a got a second request")
	}
	if g.allow("b", now) {
		t.Error("client b was charged for client a")
	}
}

// TestSweepForgetsIdleClientsAndKeepsBusyOnes covers the leak: the map is keyed
// by address, so on a reachable interface it would otherwise grow with every
// address that has ever connected, at one packet per key.
func TestSweepForgetsIdleClientsAndKeepsBusyOnes(t *testing.T) {
	g := NewGuard("", 10, 2)
	start := time.Now()
	later := start.Add(time.Hour)

	g.allow("idle", start) // last seen an hour before the sweep
	g.allow("busy", later) // last seen at the sweep

	dropped := g.Sweep(later, time.Minute)
	if dropped != 1 {
		t.Errorf("swept %d buckets, wanted 1", dropped)
	}

	g.mu.Lock()
	_, idleKept := g.buckets["idle"]
	_, busyKept := g.buckets["busy"]
	g.mu.Unlock()

	if idleKept {
		t.Error("a bucket idle for an hour survived the sweep")
	}
	if !busyKept {
		t.Error("a bucket in use was forgotten, which refunds whatever it had spent")
	}

	// The bug this caught the first time: the sweep used to read the stored
	// token count and require it to be full, so a client that spent everything
	// and then went quiet was never forgotten. That is the leak, kept alive by
	// exactly the clients most likely to have caused it. What matters is whether
	// the bucket *would* be full by now.
	g2 := NewGuard("", 10, 2)
	g2.allow("spent", start)
	g2.allow("spent", start) // empty
	if dropped := g2.Sweep(start.Add(time.Hour), time.Minute); dropped != 1 {
		t.Errorf("an emptied bucket idle for an hour was not swept (%d dropped)", dropped)
	}
}

// TestTheRateLimitSurfacesAsFourTwentyNine checks the wire behaviour, not just
// the accounting: a client that is being limited needs to be told to come back
// rather than to retry immediately.
func TestTheRateLimitSurfacesAsFourTwentyNine(t *testing.T) {
	h := NewGuard("", 1, 1).Wrap(okHandler())

	if code := request(t, h, "", "10.0.0.1:1234"); code != http.StatusOK {
		t.Fatalf("the first request got %d", code)
	}

	r := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	r.RemoteAddr = "10.0.0.1:1234"
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)

	if w.Code != http.StatusTooManyRequests {
		t.Errorf("the second request got %d, wanted 429", w.Code)
	}
	if w.Header().Get("Retry-After") == "" {
		t.Error("a 429 without Retry-After tells the client nothing about when to come back")
	}
}

// TestTheSamePortIsNotADifferentClient: RemoteAddr carries an ephemeral port
// that changes per connection, so keying on it would give every new connection
// a fresh allowance and no limit at all.
func TestTheSamePortIsNotADifferentClient(t *testing.T) {
	h := NewGuard("", 1, 1).Wrap(okHandler())

	if code := request(t, h, "", "10.0.0.1:1111"); code != http.StatusOK {
		t.Fatalf("the first request got %d", code)
	}
	if code := request(t, h, "", "10.0.0.1:2222"); code != http.StatusTooManyRequests {
		t.Errorf("a new source port got a fresh allowance: %d", code)
	}
}
