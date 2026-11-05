package api

import (
	"crypto/subtle"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"
)

// Guard is the two controls that stand between this server and anybody who can
// reach it: a bearer token, and a rate per client address.
//
// Neither was here for most of the project's life, and the README said instead
// that the server should not be exposed to anything. That sentence was true and
// it was not a control. A GPU is a spendable resource: sixty-four concurrent
// requests asking for the maximum token count occupy the batch for minutes, from
// one machine, and before this there was no way to say no.
//
// The zero value is a guard that allows everything, which is what a loopback
// server wants -- cmd/braid refuses to bind anywhere else without a token.
type Guard struct {
	token string

	rate  float64 // requests per second, 0 disables limiting
	burst float64

	mu      sync.Mutex
	buckets map[string]*bucket

	// failures is charged only by requests without a valid token, and it exists
	// whatever -rate says. See authFailureRate.
	failures map[string]*bucket
}

// How many wrong guesses at the token one client address gets.
//
// The general limit cannot do this job. It is off by default, and it was also
// behind the token check, so a wrong token was answered before anything had been
// spent: a guesser could try tokens as fast as the network allowed, at any -rate.
// This budget is separate and always on when there is a token, because guessing
// is not traffic a server should ever be tolerating, and tight, because nobody
// holding the right token ever spends from it.
//
// Ten, then one every five seconds. A client configured with an old token sees a
// few 401s and then 429s, which is the same message and costs it nothing it had.
const (
	authFailureBurst      = 10
	authFailureRate       = 0.2
	authFailureRetryAfter = "5"
)

// bucket is one client's allowance, as a token bucket rather than a window.
//
// A fixed window lets a client spend its whole allowance in the last millisecond
// of one window and again in the first of the next, which is the burst it was
// meant to prevent. A bucket refills continuously and has no edges.
type bucket struct {
	tokens float64
	last   time.Time
}

func NewGuard(token string, rate float64, burst int) *Guard {
	if burst < 1 {
		burst = 1
	}
	return &Guard{
		token:    token,
		rate:     rate,
		burst:    float64(burst),
		buckets:  make(map[string]*bucket),
		failures: make(map[string]*bucket),
	}
}

// Wrap puts the guard in front of a handler.
func (g *Guard) Wrap(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		key, now := clientKey(r), time.Now()

		// Before the token is looked at, so that a request carrying the wrong one
		// costs what any other request costs. The other order answered every wrong
		// guess for free.
		if !g.allow(key, now) {
			w.Header().Set("Retry-After", "1")
			writeError(w, http.StatusTooManyRequests, "too many requests from this client")
			return
		}

		// Also before the comparison, and for the right token too. If the right
		// token still got through, an exhausted budget would answer 429 to every
		// wrong guess and 200 to the correct one, which is not a limit on guessing
		// at all.
		if !g.mayGuess(key, now) {
			w.Header().Set("Retry-After", authFailureRetryAfter)
			writeError(w, http.StatusTooManyRequests, "too many failed attempts from this client")
			return
		}

		if !g.authorised(r) {
			g.chargeGuess(key, now)
			// No detail about why. A 401 that distinguishes "no token" from
			// "wrong token" tells somebody probing which half to work on.
			w.Header().Set("WWW-Authenticate", `Bearer realm="braid"`)
			writeError(w, http.StatusUnauthorized, "unauthorised")
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (g *Guard) authorised(r *http.Request) bool {
	if g.token == "" {
		return true
	}
	header := r.Header.Get("Authorization")
	got, found := strings.CutPrefix(header, "Bearer ")
	if !found {
		return false
	}
	// Constant time, because a byte-at-a-time comparison leaks the token one
	// character per round trip to anybody willing to make enough of them.
	return subtle.ConstantTimeCompare([]byte(got), []byte(g.token)) == 1
}

// allow spends one token from a client's bucket, refilling it for the time that
// has passed. `now` is a parameter so the test does not have to sleep.
func (g *Guard) allow(key string, now time.Time) bool {
	if g.rate <= 0 {
		return true
	}

	g.mu.Lock()
	defer g.mu.Unlock()

	b, seen := g.buckets[key]
	if !seen {
		// A new client starts full, so a first request is never rejected for
		// being first.
		b = &bucket{tokens: g.burst, last: now}
		g.buckets[key] = b
	} else {
		b.refill(now, g.rate, g.burst)
	}

	if b.tokens < 1 {
		return false
	}
	b.tokens--
	return true
}

func (b *bucket) refill(now time.Time, rate, burst float64) {
	elapsed := now.Sub(b.last).Seconds()
	if elapsed > 0 {
		b.tokens = min(burst, b.tokens+elapsed*rate)
		b.last = now
	}
}

// mayGuess reports whether a client still has failed attempts to spend. It does
// not spend one, because whether this attempt fails is not known yet, and it
// creates nothing: a client that has never failed has no entry and should not
// get one for presenting the right token.
func (g *Guard) mayGuess(key string, now time.Time) bool {
	if g.token == "" {
		return true
	}

	g.mu.Lock()
	defer g.mu.Unlock()

	b, seen := g.failures[key]
	if !seen {
		return true
	}
	b.refill(now, authFailureRate, authFailureBurst)
	return b.tokens >= 1
}

// chargeGuess records one failed attempt.
func (g *Guard) chargeGuess(key string, now time.Time) {
	g.mu.Lock()
	defer g.mu.Unlock()

	b, seen := g.failures[key]
	if !seen {
		b = &bucket{tokens: authFailureBurst, last: now}
		g.failures[key] = b
	} else {
		b.refill(now, authFailureRate, authFailureBurst)
	}
	b.tokens--
}

// Sweep drops buckets nobody has used for a while.
//
// Without it the map is a slow leak keyed by whatever addresses have ever
// connected, which on a public interface is a memory exhaustion vector with no
// request cost to the attacker: one packet per key. A full bucket is
// indistinguishable from a bucket that does not exist, so forgetting it is free.
func (g *Guard) Sweep(now time.Time, idle time.Duration) int {
	g.mu.Lock()
	defer g.mu.Unlock()

	return forget(g.buckets, now, idle, g.rate, g.burst) +
		forget(g.failures, now, idle, authFailureRate, authFailureBurst)
}

func forget(buckets map[string]*bucket, now time.Time, idle time.Duration, rate, burst float64) int {
	dropped := 0
	for key, b := range buckets {
		elapsed := now.Sub(b.last)
		if elapsed <= idle {
			continue
		}
		// The refill the elapsed time has already earned, not the stale count.
		// Reading b.tokens directly would mean a bucket that spent everything
		// and then went quiet for an hour is never forgotten -- which is the
		// leak this exists to close, kept alive by the clients most likely to
		// have caused it.
		if b.tokens+elapsed.Seconds()*rate >= burst {
			delete(buckets, key)
			dropped++
		}
	}
	return dropped
}

// clientKey is the address the limit applies to.
//
// RemoteAddr and not X-Forwarded-For: a header the client controls is a rate
// limit the client controls. Behind a proxy this limits the proxy, which is
// wrong but is wrong in the safe direction; getting it right needs the operator
// to say which proxies are trusted, and that is a flag this does not have yet.
func clientKey(r *http.Request) string {
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}
