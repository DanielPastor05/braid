package backend

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"sync"
	"sync/atomic"
	"time"
)

// Pool is a Backend over several worker processes, any of which may die.
//
// It needs no cooperation from the scheduler, and that is not luck. A sequence's
// history lives in the scheduler and every step sends the whole window, so a
// worker holds no state between calls: no cache to rebuild, no session to
// migrate, nothing that makes one worker the only one that can answer. A step
// that fails on one is simply asked of the next, with the same bytes, and the
// caller cannot tell the difference.
//
// That property is what makes failover a dozen lines here instead of a design
// problem. It is also why it is worth saying out loud: a server that kept the
// context on the GPU would have to choose between rebuilding it elsewhere and
// failing the request.
type Pool struct {
	exePath string
	prefix  string
	opts    WorkerOptions
	log     *slog.Logger

	alphabet []byte
	index    [256]int32
	seqLen   int

	mu      sync.Mutex
	slots   []*Worker // nil where a worker died and has not been replaced
	current int       // where the next Step starts looking
	retired Timings   // what the replaced workers did before they went

	failovers atomic.Int64
	restarts  atomic.Int64
	deaths    atomic.Int64
}

// NewPool starts n workers. It fails if none of them start; a pool that comes up
// short but not empty logs the shortfall and serves on what it has, because a
// server that refuses to start over one bad worker is less available than the
// thing it was protecting against.
func NewPool(exePath, prefix string, n int, opts WorkerOptions, log *slog.Logger) (*Pool, error) {
	if n < 1 {
		return nil, fmt.Errorf("backend: a pool needs at least one worker, got %d", n)
	}

	alphabet, err := os.ReadFile(prefix + ".vocab")
	if err != nil {
		return nil, fmt.Errorf("reading the alphabet: %w", err)
	}
	if len(alphabet) == 0 {
		return nil, fmt.Errorf("the alphabet in %s.vocab is empty", prefix)
	}

	p := &Pool{
		exePath:  exePath,
		prefix:   prefix,
		opts:     opts,
		log:      log,
		alphabet: alphabet,
		seqLen:   workerSeqLen,
		slots:    make([]*Worker, n),
	}
	for i := range p.index {
		p.index[i] = -1
	}
	for id, symbol := range alphabet {
		p.index[symbol] = int32(id)
	}

	var started int
	for i := range p.slots {
		w, err := NewWorker(exePath, prefix, opts, log)
		if err != nil {
			log.Error("a worker would not start", "slot", i, "error", err)
			continue
		}
		p.slots[i] = w
		started++
	}
	if started == 0 {
		return nil, fmt.Errorf("backend: no worker in the pool would start")
	}
	if started < n {
		log.Warn("the pool came up short", "wanted", n, "started", started)
	}
	return p, nil
}

func (p *Pool) SeqLen() int    { return p.seqLen }
func (p *Pool) VocabSize() int { return len(p.alphabet) }

func (p *Pool) Encode(text string) []int32 {
	ids := make([]int32, 0, len(text))
	for _, b := range []byte(text) {
		if id := p.index[b]; id >= 0 {
			ids = append(ids, id)
		}
	}
	return ids
}

func (p *Pool) Decode(ids []int32) string {
	out := make([]byte, 0, len(ids))
	for _, id := range ids {
		if id >= 0 && int(id) < len(p.alphabet) {
			out = append(out, p.alphabet[id])
		}
	}
	return string(out)
}

// Step asks one worker, and on failure asks the next, until a worker answers or
// there is none left to ask.
//
// Each worker enforces its own step timeout, so a hung one costs that timeout
// and then becomes a death like any other: killed, failed over, restarted. The
// caller's context bounds the whole sequence of attempts rather than any single
// one, which is why a hang is transparent when there is somebody to fail over
// to and fatal to the batch when there is not.
func (p *Pool) Step(ctx context.Context, windows [][]int32, lengths []int32,
	temperatures []float32, seeds []uint64) ([]int32, error) {
	var first error

	for range len(p.slots) {
		// A caller that has given up takes the whole pool with it: retrying a
		// step nobody is waiting for only costs the workers still alive.
		if err := ctx.Err(); err != nil {
			if first == nil {
				first = err
			}
			break
		}
		w, slot := p.pick()
		if w == nil {
			break
		}

		out, err := w.Step(ctx, windows, lengths, temperatures, seeds)
		if err == nil {
			return out, nil
		}

		// The window that failed is unchanged and goes to the next worker as
		// it stands, lengths and all. Nothing about this batch was consumed by
		// the attempt, which is the property that makes failover a retry.
		if first == nil {
			first = err
		}
		p.failovers.Add(1)
		p.retire(slot, w, err)
	}

	if first == nil {
		return nil, fmt.Errorf("backend: the pool has no live worker")
	}
	return nil, fmt.Errorf("backend: every worker in the pool refused the step, first was: %w", first)
}

// pick returns the worker to try, and the slot it came from.
func (p *Pool) pick() (*Worker, int) {
	p.mu.Lock()
	defer p.mu.Unlock()

	for range p.slots {
		slot := p.current
		p.current = (p.current + 1) % len(p.slots)
		if w := p.slots[slot]; w != nil {
			return w, slot
		}
	}
	return nil, -1
}

// retire empties a slot and starts filling it again, without waiting. The step
// that discovered the death is already on its way to another worker, and a
// process start is tens of milliseconds it should not be holding.
func (p *Pool) retire(slot int, w *Worker, cause error) {
	p.mu.Lock()
	if p.slots[slot] != w {
		p.mu.Unlock() // somebody else already retired it
		return
	}
	p.slots[slot] = nil
	// Its counters go into the pool's running total before the object does,
	// so a restart does not silently reset the timing history.
	t := w.Timings()
	p.retired.Steps += t.Steps
	p.retired.Sequences += t.Sequences
	p.retired.Wall += t.Wall
	p.retired.Build += t.Build
	p.retired.Forward += t.Forward
	p.retired.Copy += t.Copy
	p.retired.Sample += t.Sample
	p.retired.Kernels += t.Kernels
	p.retired.ToDevice += t.ToDevice
	p.retired.ToHost += t.ToHost
	p.mu.Unlock()

	p.deaths.Add(1)
	p.log.Warn("a worker died and was taken out of the pool", "slot", slot, "error", cause)
	_ = w.Close()

	go p.refill(slot)
}

// refill keeps trying to put a worker back, backing off so that a permanently
// broken binary does not become a spawn loop.
func (p *Pool) refill(slot int) {
	backoff := 100 * time.Millisecond
	for attempt := 1; ; attempt++ {
		p.mu.Lock()
		closed := p.slots == nil
		p.mu.Unlock()
		if closed {
			return
		}

		w, err := NewWorker(p.exePath, p.prefix, p.opts, p.log)
		if err == nil {
			p.mu.Lock()
			if p.slots == nil { // closed while we were starting
				p.mu.Unlock()
				_ = w.Close()
				return
			}
			p.slots[slot] = w
			p.mu.Unlock()

			p.restarts.Add(1)
			p.log.Info("a worker rejoined the pool", "slot", slot, "attempt", attempt)
			return
		}

		p.log.Error("could not restart a worker", "slot", slot, "attempt", attempt, "error", err)
		time.Sleep(backoff)
		if backoff < 5*time.Second {
			backoff *= 2
		}
	}
}

// PoolStats is what the pool did, as opposed to what the model did.
type PoolStats struct {
	Workers   int   `json:"workers"`
	Live      int   `json:"live"`
	Deaths    int64 `json:"deaths"`
	Restarts  int64 `json:"restarts"`
	Failovers int64 `json:"failovers"`
}

func (p *Pool) PoolStats() PoolStats {
	p.mu.Lock()
	live := 0
	for _, w := range p.slots {
		if w != nil {
			live++
		}
	}
	total := len(p.slots)
	p.mu.Unlock()

	return PoolStats{
		Workers:   total,
		Live:      live,
		Deaths:    p.deaths.Load(),
		Restarts:  p.restarts.Load(),
		Failovers: p.failovers.Load(),
	}
}

// Timings sums the live workers and everything the replaced ones did.
func (p *Pool) Timings() Timings {
	p.mu.Lock()
	t := p.retired
	for _, w := range p.slots {
		if w == nil {
			continue
		}
		u := w.Timings()
		t.Steps += u.Steps
		t.Sequences += u.Sequences
		t.Wall += u.Wall
		t.Build += u.Build
		t.Forward += u.Forward
		t.Copy += u.Copy
		t.Sample += u.Sample
		t.Kernels += u.Kernels
		t.ToDevice += u.ToDevice
		t.ToHost += u.ToHost
	}
	p.mu.Unlock()

	if t.Steps == 0 {
		return t
	}
	per := func(d time.Duration) float64 {
		return float64(d.Microseconds()) / 1000 / float64(t.Steps)
	}
	t.WallMS = per(t.Wall)
	t.BuildMS = per(t.Build)
	t.ForwardMS = per(t.Forward)
	t.CopyMS = per(t.Copy)
	t.SampleMS = per(t.Sample)
	t.PipeMS = t.WallMS - t.BuildMS - t.ForwardMS - t.CopyMS - t.SampleMS
	if t.WallMS > 0 {
		t.PipeShare = t.PipeMS / t.WallMS
	}
	t.KernelsPerStep = float64(t.Kernels) / float64(t.Steps)
	t.ToDevicePerStep = float64(t.ToDevice) / float64(t.Steps)
	t.ToHostPerStep = float64(t.ToHost) / float64(t.Steps)
	return t
}

// Pids reports the live workers' process ids, so that something outside can go
// and kill one. That is what the chaos test does, and it is the only honest way
// to test a death: signalling a worker to stop politely would exercise the
// shutdown path, which is not the path in question.
func (p *Pool) Pids() []int {
	p.mu.Lock()
	defer p.mu.Unlock()

	out := make([]int, 0, len(p.slots))
	for _, w := range p.slots {
		if w != nil {
			if pid := w.Pid(); pid > 0 {
				out = append(out, pid)
			}
		}
	}
	return out
}

// Close stops every worker and stops the refill goroutines from putting more
// back.
func (p *Pool) Close() error {
	p.mu.Lock()
	slots := p.slots
	p.slots = nil // the signal refill watches for
	p.mu.Unlock()

	var first error
	for _, w := range slots {
		if w == nil {
			continue
		}
		if err := w.Close(); err != nil && first == nil {
			first = err
		}
	}
	return first
}
