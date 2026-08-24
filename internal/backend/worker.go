package backend

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"log/slog"
	"math"
	"os"
	"os/exec"
	"sync"
	"time"
)

// Worker is a Backend backed by a braid_worker process holding the model.
//
// The process boundary is not isolation for its own sake. nvcc on Windows
// compiles only through MSVC and cgo links only through a GCC-compatible
// toolchain, so the engine cannot be linked into this binary at all; a pipe is
// the one interface both toolchains agree on. Everything that follows from that
// -- a worker that can die without taking the server with it, and later more
// than one of them -- is a consequence worth having.
type Worker struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Reader

	alphabet []byte
	index    [256]int32 // byte to id, -1 where the alphabet has no such byte
	seqLen   int

	// One request buffer, reused. Step is called from the scheduler's single
	// goroutine, so the mutex guards only against a Close racing a Step.
	mu     sync.Mutex
	frame  []byte
	result []byte
	closed bool

	timing Timings
}

// Timings is where a step's wall time went, summed over every step so far.
//
// Inside is what the worker measured of itself; Wall is what this side measured
// around the whole call. Everything between the two is the pipe: two writes,
// two reads, and the serialising at each end. Reporting it as a subtraction
// rather than as its own measurement is deliberate -- it means nothing can hide
// in the gap, because the gap is the answer.
type Timings struct {
	Steps     int64         `json:"steps"`
	Sequences int64         `json:"sequences"`
	Wall      time.Duration `json:"-"`
	Build     time.Duration `json:"-"`
	Forward   time.Duration `json:"-"`
	Sample    time.Duration `json:"-"`
	Kernels   int64         `json:"-"`

	WallMS    float64 `json:"wall_ms_per_step"`
	BuildMS   float64 `json:"build_ms_per_step"`
	ForwardMS float64 `json:"forward_ms_per_step"`
	SampleMS  float64 `json:"sample_ms_per_step"`
	PipeMS    float64 `json:"pipe_ms_per_step"`
	PipeShare float64 `json:"pipe_share"`

	// KernelsPerStep is how many CUDA kernels a forward launched, on average.
	// Zero means the work fell below the engine's size threshold and ran on the
	// CPU -- which the engine does silently and on purpose, and which is
	// otherwise invisible in a timing that only got slower.
	KernelsPerStep float64 `json:"kernels_per_step"`
}

const (
	frameMagic  uint32 = 0x33445242 // 'BRD3'
	statusOK    uint32 = 0
	statusError uint32 = 1

	// workerSeqLen must match braid::kSeqLen in engine/charmodel.hpp. It is not
	// negotiated over the pipe because a disagreement would show up as garbled
	// text rather than an error, and a constant in two files that must match is
	// at least greppable.
	workerSeqLen = 64
)

// WorkerOptions tunes the engine inside the worker.
//
// Both thresholds are the engine's own: below them an operation stays on the
// CPU, because for a training step the transfer costs more than the arithmetic
// saves. Serving is not a training step -- at this model's size a batch of one
// falls under the line and never touches the card at all -- so they are exposed
// here rather than left at whatever the engine was tuned for. Zero means leave
// the engine's default alone.
type WorkerOptions struct {
	MinMatmulFlops uint64
	MinElements    uint64
}

// NewWorker starts the worker process and loads the alphabet the model was
// trained on. The checkpoint prefix names two files: prefix.bin and
// prefix.vocab, both written by braid_train.
func NewWorker(exePath, prefix string, opts WorkerOptions, log *slog.Logger) (*Worker, error) {
	alphabet, err := os.ReadFile(prefix + ".vocab")
	if err != nil {
		return nil, fmt.Errorf("reading the alphabet: %w", err)
	}
	if len(alphabet) == 0 {
		return nil, fmt.Errorf("the alphabet in %s.vocab is empty", prefix)
	}

	w := &Worker{alphabet: alphabet, seqLen: workerSeqLen}
	for i := range w.index {
		w.index[i] = -1
	}
	for id, symbol := range alphabet {
		w.index[symbol] = int32(id)
	}

	w.cmd = exec.Command(exePath, prefix)

	// The engine reads these once, when its threshold globals are initialised,
	// so they have to be in the environment before the process starts. Passing
	// them this way rather than adding a flag to the worker keeps one mechanism
	// instead of two: the engine already documents these names.
	w.cmd.Env = os.Environ()
	if opts.MinMatmulFlops > 0 {
		w.cmd.Env = append(w.cmd.Env,
			fmt.Sprintf("ENGINE_CUDA_MIN_FLOPS=%d", opts.MinMatmulFlops))
	}
	if opts.MinElements > 0 {
		w.cmd.Env = append(w.cmd.Env,
			fmt.Sprintf("ENGINE_CUDA_MIN_ELEMENTS=%d", opts.MinElements))
	}

	stdin, err := w.cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("worker stdin: %w", err)
	}
	stdout, err := w.cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("worker stdout: %w", err)
	}
	stderr, err := w.cmd.StderrPipe()
	if err != nil {
		return nil, fmt.Errorf("worker stderr: %w", err)
	}

	if err := w.cmd.Start(); err != nil {
		return nil, fmt.Errorf("starting %s: %w", exePath, err)
	}

	// The worker announces its device on stderr and reports kernel counts when
	// it stops. Forwarding it means a fallback to the CPU path is visible in
	// the server's log rather than only in the numbers.
	go func() {
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			log.Info("worker", "line", scanner.Text())
		}
	}()

	w.stdin = stdin
	w.stdout = bufio.NewReaderSize(stdout, 1<<16)
	return w, nil
}

func (w *Worker) SeqLen() int    { return w.seqLen }
func (w *Worker) VocabSize() int { return len(w.alphabet) }

func (w *Worker) Encode(text string) []int32 {
	ids := make([]int32, 0, len(text))
	for _, b := range []byte(text) {
		// Characters the model was never trained on are dropped rather than
		// mapped to a substitute, which is what the engine's CharVocab does.
		// A prompt of nothing but unknown bytes encodes to nothing, and the
		// model generates from an empty context, which is a real answer.
		if id := w.index[b]; id >= 0 {
			ids = append(ids, id)
		}
	}
	return ids
}

func (w *Worker) Decode(ids []int32) string {
	out := make([]byte, 0, len(ids))
	for _, id := range ids {
		if id >= 0 && int(id) < len(w.alphabet) {
			out = append(out, w.alphabet[id])
		}
	}
	return string(out)
}

func (w *Worker) Step(windows [][]int32, temperatures []float32, seeds []uint64) ([]int32, error) {
	if len(windows) != len(temperatures) || len(windows) != len(seeds) {
		return nil, errRagged
	}
	if len(windows) == 0 {
		return nil, fmt.Errorf("backend: a step with no sequences in it")
	}
	for _, window := range windows {
		if len(window) != w.seqLen {
			return nil, errWindowWidth
		}
	}

	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return nil, fmt.Errorf("backend: the worker is closed")
	}

	// Started here rather than at the top of the function: what the scheduler
	// pays for a step is the serialising, the pipe and the parsing, and the
	// argument checks above are not part of that.
	started := time.Now()

	n := len(windows)
	size := 4 + 4 + n*w.seqLen*4 + n*4 + n*8
	if cap(w.frame) < size {
		w.frame = make([]byte, size)
	}
	frame := w.frame[:size]

	binary.LittleEndian.PutUint32(frame[0:], frameMagic)
	binary.LittleEndian.PutUint32(frame[4:], uint32(n))
	at := 8
	for _, window := range windows {
		for _, id := range window {
			binary.LittleEndian.PutUint32(frame[at:], uint32(id))
			at += 4
		}
	}
	for _, t := range temperatures {
		binary.LittleEndian.PutUint32(frame[at:], math.Float32bits(t))
		at += 4
	}
	for _, seed := range seeds {
		binary.LittleEndian.PutUint64(frame[at:], seed)
		at += 8
	}

	if _, err := w.stdin.Write(frame); err != nil {
		return nil, fmt.Errorf("backend: writing a step to the worker: %w", err)
	}

	var status uint32
	if err := binary.Read(w.stdout, binary.LittleEndian, &status); err != nil {
		return nil, fmt.Errorf("backend: reading the worker's status: %w", err)
	}
	if status == statusError {
		var length uint32
		if err := binary.Read(w.stdout, binary.LittleEndian, &length); err != nil {
			return nil, fmt.Errorf("backend: reading the worker's error length: %w", err)
		}
		message := make([]byte, length)
		if _, err := io.ReadFull(w.stdout, message); err != nil {
			return nil, fmt.Errorf("backend: reading the worker's error: %w", err)
		}
		return nil, fmt.Errorf("backend: the worker refused the step: %s", message)
	}
	if status != statusOK {
		return nil, fmt.Errorf("backend: the worker sent status %d, which is not a status", status)
	}

	// n ids, the three timings the worker measured of itself, and the count
	// of kernels its forward launched.
	const timingBytes = 4 * 8
	need := n*4 + timingBytes
	if cap(w.result) < need {
		w.result = make([]byte, need)
	}
	result := w.result[:need]
	if _, err := io.ReadFull(w.stdout, result); err != nil {
		return nil, fmt.Errorf("backend: reading %d ids from the worker: %w", n, err)
	}

	out := make([]int32, n)
	for i := range out {
		out[i] = int32(binary.LittleEndian.Uint32(result[i*4:]))
	}

	at = n * 4
	w.timing.Steps++
	w.timing.Sequences += int64(n)
	w.timing.Wall += time.Since(started)
	w.timing.Build += time.Duration(binary.LittleEndian.Uint64(result[at:]))
	w.timing.Forward += time.Duration(binary.LittleEndian.Uint64(result[at+8:]))
	w.timing.Sample += time.Duration(binary.LittleEndian.Uint64(result[at+16:]))
	w.timing.Kernels += int64(binary.LittleEndian.Uint64(result[at+24:]))

	return out, nil
}

// Timings reports where the steps taken so far spent their time.
func (w *Worker) Timings() Timings {
	w.mu.Lock()
	defer w.mu.Unlock()

	t := w.timing
	if t.Steps == 0 {
		return t
	}

	per := func(d time.Duration) float64 {
		return float64(d.Microseconds()) / 1000 / float64(t.Steps)
	}
	t.WallMS = per(t.Wall)
	t.BuildMS = per(t.Build)
	t.ForwardMS = per(t.Forward)
	t.SampleMS = per(t.Sample)
	t.KernelsPerStep = float64(t.Kernels) / float64(t.Steps)

	// What is left when the worker's own accounting is taken off the wall
	// clock. It can come out slightly negative on a machine whose two clocks
	// disagree at the microsecond, and it is reported as measured rather than
	// clamped, because a negative here means the measurement is wrong and that
	// is worth seeing.
	t.PipeMS = t.WallMS - t.BuildMS - t.ForwardMS - t.SampleMS
	if t.WallMS > 0 {
		t.PipeShare = t.PipeMS / t.WallMS
	}
	return t
}

// Close shuts the worker down by closing its stdin, which the protocol defines
// as a clean stop, and then waits for it to exit.
func (w *Worker) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return nil
	}
	w.closed = true

	if err := w.stdin.Close(); err != nil {
		_ = w.cmd.Process.Kill()
		return fmt.Errorf("backend: closing the worker's stdin: %w", err)
	}
	if err := w.cmd.Wait(); err != nil {
		return fmt.Errorf("backend: the worker exited badly: %w", err)
	}
	return nil
}
