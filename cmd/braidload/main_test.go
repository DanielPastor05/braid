package main

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func answering(status int, body string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
}

func TestHealthIsOnlyHealthyOnAnOK(t *testing.T) {
	for _, c := range []struct {
		status int
		ok     bool
	}{
		{http.StatusOK, true},
		{http.StatusUnauthorized, false},
		{http.StatusServiceUnavailable, false},
	} {
		srv := answering(c.status, `{}`)
		err := checkHealth(srv.Client(), srv.URL)
		srv.Close()
		if (err == nil) != c.ok {
			t.Errorf("status %d: got error %v", c.status, err)
		}
	}
}

func TestHealthOfAServerThatIsNotThere(t *testing.T) {
	srv := answering(http.StatusOK, `{}`)
	client, addr := srv.Client(), srv.URL
	srv.Close()

	if err := checkHealth(client, addr); err == nil {
		t.Error("a closed server was reported healthy")
	}
}

// TestStatsRefuseAnErrorBody: the guard answers 401 with valid JSON, and decoding
// that into stats succeeds with every field at zero. The harness would then have
// measured a server it was never allowed to talk to.
func TestStatsRefuseAnErrorBody(t *testing.T) {
	srv := answering(http.StatusUnauthorized, `{"error":"unauthorised"}`)
	defer srv.Close()

	if _, err := readStats(srv.Client(), srv.URL); err == nil {
		t.Error("a 401 from /stats was read as stats")
	}
}

func TestBearerSendsTheToken(t *testing.T) {
	var got string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r.Header.Get("Authorization")
	}))
	defer srv.Close()

	client := &http.Client{Transport: bearer{token: "s3cret", next: http.DefaultTransport}}
	resp, err := client.Get(srv.URL)
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	resp.Body.Close()

	if got != "Bearer s3cret" {
		t.Errorf("Authorization was %q", got)
	}
}
