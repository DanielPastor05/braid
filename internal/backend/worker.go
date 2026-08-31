package backend

import (
	"bufio"
	"context"
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

	stepTimeout time.Duration
	timing      Timings

	// A diagnostic, off unless asked for. See EmitLogits.
	emitLogits bool
	lastLogits []float32
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
	Copy      time.Duration `json:"-"`
	Sample    time.Duration `json:"-"`
	Kernels   int64         `json:"-"`
	ToDevice  int64         `json:"-"`
	ToHost    int64         `json:"-"`

	WallMS    float64 `json:"wall_ms_per_step"`
	BuildMS   float64 `json:"build_ms_per_step"`
	ForwardMS float64 `json:"forward_ms_per_step"`

	// CopyMS is the device-to-host transfer of the logits, separated from the
	// model's own time because the worker pulls back every position of the
	// window when the sampling reads only the last one of each sequence.
	CopyMS    float64 `json:"copy_ms_per_step"`
	SampleMS  float64 `json:"sample_ms_per_step"`
	PipeMS    float64 `json:"pipe_ms_per_step"`
	PipeShare float64 `json:"pipe_share"`

	// KernelsPerStep is how many CUDA kernels a forward launched, on average.
	// Zero means the work fell below the engine's size threshold and ran on the
	// CPU -- which the engine does silently and on purpose, and which is
	// otherwise invisible in a timing that only got slower.
	KernelsPerStep float64 `json:"kernels_per_step"`

	// PCIe crossings per step. An operation the engine keeps on the host still
	// has to get its inputs there and its result back, so these rise as the
	// kernel count falls -- which is how a drop in kernels is told apart from a
	// change in how the same work is launched.
	ToDevicePerStep float64 `json:"to_device_per_step"`
	ToHostPerStep   float64 `json:"to_host_per_step"`
}

const (
	frameMagic  uint32 = 0x36445242 // 'BRD6'
	statusOK    uint32 = 0
	statusError uint32 = 1

	// workerSeqLen must match braid::kSeqLen in engine/charmodel.hpp. It is not
	// negotiated over the pipe because a disagreement would show up as garbled
	// text rather than an error, and a constant in two files that must match is
	// at least greppable.
	//
	// TestTheWorkerAgreesAboutTheWindow is what makes the greppability
	// unnecessary: it asks a real worker and fails if the two ever part. Growing
	// the model from a 64-id context to 256 is exactly the change that would
	// have broken this quietly.
	workerSeqLen = 1024

	// defaultStepTimeout is four orders of magnitude above a real step, so it
	// fires only for a process that has stopped answering rather than one
	// having a bad day.
	defaultStepTimeout = 30 * time.Second

	// maxErrorMessage caps the error text a worker may announce before the
	// server allocates room for it. The worker writes the length and the server
	// believes it, so without a bound the number is a request to allocate --
	// four bytes read at the wrong offset ask for four gigabytes.
	maxErrorMessage = 64 << 10

	// reapGrace bounds every wait on a process that is supposed to be dying.
	// The first version of the timeout fix waited on the reaper without one,
	// which fixed the hang for the case where the kill lands and reproduced it
	// exactly for the case where it does not.
	reapGrace = 5 * time.Second
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

	// StepTimeout is how long one step may take before the worker is presumed
	// hung and killed. Zero means defaultStepTimeout. A step is milliseconds;
	// this is not a latency budget, it is the line between slow and gone.
	StepTimeout time.Duration

	// EmitLogits asks the worker to append the logits of every sampled row to
	// each response.
	//
	// It is off by default and serving does not want it: n * vocab floats a step
	// that nobody reads, when the sampled id is the whole answer. What wants it
	// is the batch-invariance measurement. That test compares the *token* two
	// runs produced, so it can only see a difference in the arithmetic when the
	// difference happens to cross a boundary of the sampler's inverse CDF --
	// which measures the probability that the noise mattered rather than the
	// noise. With the logits it measures the noise.
	EmitLogits bool

	// MinLayerNormElements is the third of the engine's thresholds, and the one
	// that decided whether a small forward chained across the card or came home
	// at every normalisation.
	//
	// It mattered enormously to the 172 728-parameter model, whose LayerNorm saw
	// n*6144 elements and so did not clear the engine's 2^15 floor until a batch
	// of six. At 384 wide over a 256-id context that is n*98304, which clears it
	// at a batch of one -- so the knob is inert here and kept only because a
	// smaller model would want it again. What the README says about it is
	// history, and labelled as such.
	MinLayerNormElements uint64
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

	timeout := opts.StepTimeout
	if timeout <= 0 {
		timeout = defaultStepTimeout
	}
	w := &Worker{
		alphabet:    alphabet,
		seqLen:      workerSeqLen,
		stepTimeout: timeout,
		emitLogits:  opts.EmitLogits,
	}
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
	if opts.MinLayerNormElements > 0 {
		w.cmd.Env = append(w.cmd.Env,
			fmt.Sprintf("ENGINE_CUDA_MIN_LAYERNORM=%d", opts.MinLayerNormElements))
	}
	if opts.EmitLogits {
		w.cmd.Env = append(w.cmd.Env, "BRAID_EMIT_LOGITS=1")
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

// Pid is the worker process's id, or 0 once it has been closed. The pool hands
// these out so that a test can kill one the way the operating system would.
func (w *Worker) Pid() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed || w.cmd == nil || w.cmd.Process == nil {
		return 0
	}
	return w.cmd.Process.Pid
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

func (w *Worker) Step(ctx context.Context, windows [][]int32, lengths []int32,
	temperatures []float32, seeds []uint64) ([]int32, error) {
	if len(windows) != len(temperatures) || len(windows) != len(seeds) ||
		len(windows) != len(lengths) {
		return nil, errRagged
	}
	if len(windows) == 0 {
		return nil, fmt.Errorf("backend: a step with no sequences in it")
	}
	for i, window := range windows {
		if len(window) != w.seqLen {
			return nil, errWindowWidth
		}
		// Caught here rather than in the worker, where it would be a killed
		// process and a failover instead of an error naming the row.
		if lengths[i] < 1 || int(lengths[i]) > w.seqLen {
			return nil, errWindowLength
		}
	}

	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return nil, fmt.Errorf("backend: the worker is closed")
	}

	// The worker's own ceiling, applied whether or not the caller brought one.
	// A step is milliseconds; anything approaching this is a process that has
	// stopped answering rather than one that is being slow.
	ctx, cancel := context.WithTimeout(ctx, w.stepTimeout)
	defer cancel()

	// Started here rather than at the top of the function: what the scheduler
	// pays for a step is the serialising, the pipe and the parsing, and the
	// argument checks above are not part of that.
	started := time.Now()

	n := len(windows)
	size := 4 + 4 + n*w.seqLen*4 + n*4 + n*4 + n*8
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
	for _, length := range lengths {
		binary.LittleEndian.PutUint32(frame[at:], uint32(length))
		at += 4
	}
	for _, t := range temperatures {
		binary.LittleEndian.PutUint32(frame[at:], math.Float32bits(t))
		at += 4
	}
	for _, seed := range seeds {
		binary.LittleEndian.PutUint64(frame[at:], seed)
		at += 8
	}

	// The exchange runs on another goroutine so that the deadline can be
	// enforced at all: a pipe read has no timeout of its own, and the process
	// at the other end may be alive and simply not answering.
	type answer struct {
		ids     []int32
		timings [7]uint64
		err     error
	}
	done := make(chan answer, 1)
	go func() {
		ids, timings, err := w.exchange(frame, n)
		done <- answer{ids, timings, err}
	}()

	select {
	case got := <-done:
		if got.err != nil {
			return nil, got.err
		}
		w.timing.Steps++
		w.timing.Sequences += int64(n)
		w.timing.Wall += time.Since(started)
		w.timing.Build += time.Duration(got.timings[0])
		w.timing.Forward += time.Duration(got.timings[1])
		w.timing.Copy += time.Duration(got.timings[2])
		w.timing.Sample += time.Duration(got.timings[3])
		w.timing.Kernels += int64(got.timings[4])
		w.timing.ToDevice += int64(got.timings[5])
		w.timing.ToHost += int64(got.timings[6])
		return got.ids, nil

	case <-ctx.Done():
		// Killing it is not a choice. The read is blocked in the kernel with no
		// message that means "never mind", and a worker that answered late
		// would answer the *next* step with this step's ids -- the protocol has
		// no request identifier and no way to notice. So the process goes, the
		// goroutine unblocks on the broken pipe, and a pool treats it as the
		// death it now is.
		w.killLocked()

		// Bounded, because the reap is the same unbounded wait this whole
		// change exists to remove -- one layer down. If the kill somehow does
		// not land, leaking a goroutine is the cheaper failure: it holds this
		// worker's buffers, and this worker is about to be retired and never
		// stepped again, while blocking here would hold the scheduler.
		select {
		case <-done:
		case <-time.After(reapGrace):
		}
		return nil, fmt.Errorf("backend: the worker did not answer within %v: %w",
			w.stepTimeout, ctx.Err())
	}
}

// exchange writes one frame and reads its answer, blocking for as long as the
// worker takes. Bounding that is Step's business, not this function's.
func (w *Worker) exchange(frame []byte, n int) ([]int32, [7]uint64, error) {
	var timings [7]uint64

	if _, err := w.stdin.Write(frame); err != nil {
		return nil, timings, fmt.Errorf("backend: writing a step to the worker: %w", err)
	}

	var status uint32
	if err := binary.Read(w.stdout, binary.LittleEndian, &status); err != nil {
		return nil, timings, fmt.Errorf("backend: reading the worker's status: %w", err)
	}
	if status == statusError {
		var length uint32
		if err := binary.Read(w.stdout, binary.LittleEndian, &length); err != nil {
			return nil, timings, fmt.Errorf("backend: reading the worker's error length: %w", err)
		}
		// Bounded before it is allocated. The length arrives from the worker,
		// and a worker that is confused rather than malicious is enough: four
		// bytes read at the wrong offset are a four-gigabyte allocation, and the
		// server dies of a message it was trying to log. Nothing the worker has
		// to say needs more than this.
		if length > maxErrorMessage {
			return nil, timings, fmt.Errorf(
				"backend: the worker announced a %d-byte error message, which is not a message",
				length)
		}
		message := make([]byte, length)
		if _, err := io.ReadFull(w.stdout, message); err != nil {
			return nil, timings, fmt.Errorf("backend: reading the worker's error: %w", err)
		}
		return nil, timings, fmt.Errorf("backend: the worker refused the step: %s", message)
	}
	if status != statusOK {
		return nil, timings, fmt.Errorf(
			"backend: the worker sent status %d, which is not a status", status)
	}

	// n ids, the timings the worker measured of itself, the count of kernels its
	// forward launched, and -- only when it was asked for -- the logits of every
	// sampled row.
	const timingBytes = 7 * 8
	need := n*4 + timingBytes
	logitCount := 0
	if w.emitLogits {
		logitCount = n * len(w.alphabet)
		need += logitCount * 4
	}
	if cap(w.result) < need {
		w.result = make([]byte, need)
	}
	result := w.result[:need]
	if _, err := io.ReadFull(w.stdout, result); err != nil {
		return nil, timings, fmt.Errorf("backend: reading %d ids from the worker: %w", n, err)
	}

	out := make([]int32, n)
	for i := range out {
		out[i] = int32(binary.LittleEndian.Uint32(result[i*4:]))
	}
	for i := range timings {
		timings[i] = binary.LittleEndian.Uint64(result[n*4+i*8:])
	}
	if logitCount > 0 {
		at := n*4 + timingBytes
		w.lastLogits = make([]float32, logitCount)
		for i := range w.lastLogits {
			w.lastLogits[i] = math.Float32frombits(binary.LittleEndian.Uint32(result[at+i*4:]))
		}
	}
	return out, timings, nil
}

// LastLogits is the logits of every sampled row of the most recent step, laid
// out row-major as n * VocabSize, or nil unless the worker was started with
// EmitLogits.
//
// It is deliberately the *last* step rather than a return from Step: threading a
// diagnostic through the Backend interface would put it in front of every
// implementation and every caller, to be ignored by all of them.
func (w *Worker) LastLogits() []float32 {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.lastLogits
}

// killLocked ends the process. The caller holds mu; closed is deliberately left
// alone so that Close still reaps the corpse.
func (w *Worker) killLocked() {
	if w.cmd != nil && w.cmd.Process != nil {
		_ = w.cmd.Process.Kill()
	}
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
	t.CopyMS = per(t.Copy)
	t.SampleMS = per(t.Sample)
	t.KernelsPerStep = float64(t.Kernels) / float64(t.Steps)
	t.ToDevicePerStep = float64(t.ToDevice) / float64(t.Steps)
	t.ToHostPerStep = float64(t.ToHost) / float64(t.Steps)

	// What is left when the worker's own accounting is taken off the wall
	// clock. It can come out slightly negative on a machine whose two clocks
	// disagree at the microsecond, and it is reported as measured rather than
	// clamped, because a negative here means the measurement is wrong and that
	// is worth seeing.
	t.PipeMS = t.WallMS - t.BuildMS - t.ForwardMS - t.CopyMS - t.SampleMS
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
		w.killLocked()
	}

	// Closing stdin asks the worker to stop, and a worker wedged in a kernel
	// call is not reading its stdin to be asked. Waiting on that unconditionally
	// would block whoever is closing -- and retire() closes from the scheduler's
	// own goroutine, so an unbounded Wait here is the hang again by another
	// route. Ask, then insist, then give up and say so.
	done := make(chan error, 1)
	go func() { done <- w.cmd.Wait() }()

	select {
	case err := <-done:
		if err != nil {
			return fmt.Errorf("backend: the worker exited badly: %w", err)
		}
		return nil
	case <-time.After(reapGrace):
	}

	w.killLocked()
	select {
	case err := <-done:
		if err != nil {
			return fmt.Errorf("backend: the worker had to be killed to stop: %w", err)
		}
		return nil
	case <-time.After(reapGrace):
		return fmt.Errorf("backend: the worker would not exit within %v of being killed",
			reapGrace)
	}
}
