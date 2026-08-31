package backend

import (
	"encoding/binary"
	"fmt"
	"hash/fnv"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// A worker that speaks the protocol and holds no model, so that everything
// above it -- the frame, the pool, the failover, the restart -- can be tested
// without a GPU, without MSVC, and without a checkpoint.
//
// It is not a separate binary. The test binary re-executes itself with
// BRAID_FAKE_WORKER set and becomes a worker, which works because NewWorker
// hands the child os.Environ(). The parent has already passed the check in
// TestMain by the time it sets the variable, so it cannot turn into a worker
// itself.
//
// The point is that CI runs these. The tests against the real engine live
// behind BRAID_WORKER and are skipped on every machine without a card, which
// left the pool and its failover -- the part of this repository that is about
// processes rather than arithmetic -- verified nowhere but on one desk.

const fakeWorkerEnv = "BRAID_FAKE_WORKER"

// Modes, as BRAID_FAKE_WORKER values:
//
//	normal      answer every step, forever
//	die:N       answer N steps, then exit without a word -- a crash, an OOM
//	            kill, a driver reset: the process is simply gone
//	error:N     answer N steps, then return a protocol error frame and exit,
//	            which is what the real worker does: every write_error in
//	            worker.cpp is followed by `return 1`
//	garbage:N   answer N steps, then send a status that is not a status
//	slow:MS     answer every step, MS milliseconds late
//	hang:N      answer N steps, then read the next frame and never reply --
//	            alive, holding the pipe open, silent. The failure a dead
//	            process cannot simulate, because a dead process closes its
//	            pipe and the read fails immediately.
func TestMain(m *testing.M) {
	if mode := os.Getenv(fakeWorkerEnv); mode != "" {
		fakeWorkerMain(mode)
	}
	os.Exit(m.Run())
}

func fakeWorkerMain(mode string) {
	kind, arg := mode, 0
	if before, after, found := strings.Cut(mode, ":"); found {
		kind = before
		n, err := strconv.Atoi(after)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fake worker: %q is not a number in %q\n", after, mode)
			os.Exit(2)
		}
		arg = n
	}

	fmt.Fprintf(os.Stderr, "fake worker ready: mode %s\n", mode)

	for served := 0; ; served++ {
		var magic, n uint32
		if err := binary.Read(os.Stdin, binary.LittleEndian, &magic); err != nil {
			os.Exit(0) // the server closed the pipe: a clean stop
		}
		if magic != frameMagic {
			writeFakeError("bad frame magic")
			os.Exit(1)
		}
		if err := binary.Read(os.Stdin, binary.LittleEndian, &n); err != nil {
			os.Exit(1)
		}

		// The whole request has to come off the pipe before anything is decided,
		// or a misbehaviour would leave a half-read frame behind and the next
		// test would see a protocol bug that this file invented.
		// ids + lengths + slots + temperatures + seeds. The slots are read and
		// ignored: this worker has no cache, which is exactly the case the
		// protocol has to keep working -- a replacement after a failover is a
		// worker that knows nothing and recomputes.
		body := make([]byte, int(n)*workerSeqLen*4+int(n)*4+int(n)*4+int(n)*4+int(n)*8)
		if _, err := io.ReadFull(os.Stdin, body); err != nil {
			os.Exit(1)
		}

		if served >= arg {
			switch kind {
			case "die":
				os.Exit(1)
			case "error":
				writeFakeError("the fake worker was told to refuse")
				os.Exit(1)
			case "garbage":
				_ = binary.Write(os.Stdout, binary.LittleEndian, uint32(99))
				continue
			case "hang":
				// Not select{}: with no other goroutine runnable the Go
				// runtime calls that a deadlock and kills the process, which
				// produces the EOF this mode exists to avoid. Sleeping is
				// indistinguishable from a wedged GPU call and stays alive.
				time.Sleep(365 * 24 * time.Hour)
			}
		}
		if kind == "slow" {
			time.Sleep(time.Duration(arg) * time.Millisecond)
		}

		lengths := body[int(n)*workerSeqLen*4:]
		out := make([]byte, 0, int(n)*4+7*8)
		out = binary.LittleEndian.AppendUint32(out, statusOK)
		for i := range int(n) {
			window := body[i*workerSeqLen*4 : (i+1)*workerSeqLen*4]
			length := binary.LittleEndian.Uint32(lengths[i*4:])
			out = binary.LittleEndian.AppendUint32(out, uint32(fakeNextID(window, length)))
		}
		for _, v := range fakeStepReport {
			out = binary.LittleEndian.AppendUint64(out, v)
		}
		if _, err := os.Stdout.Write(out); err != nil {
			os.Exit(1)
		}
	}
}

// fakeStepReport is what every fake step claims about itself: build, forward,
// copy, sample, kernels, to_device, to_host. Fixed so a test can assert the
// numbers arrived rather than assert they are plausible.
var fakeStepReport = [7]uint64{1000, 2000, 250, 500, 60, 1, 1}

func writeFakeError(message string) {
	out := binary.LittleEndian.AppendUint32(nil, statusError)
	out = binary.LittleEndian.AppendUint32(out, uint32(len(message)))
	out = append(out, message...)
	_, _ = os.Stdout.Write(out)
}

// fakeNextID is the answer a fake gives, derived from every byte of the window
// and from the length that came with it. A hash rather than, say, the last id,
// so that a frame assembled in the wrong order or truncated produces a different
// answer instead of the right one -- and the length is in the hash because a
// real worker samples at length-1, so a length that did not survive the pipe has
// to change the answer here too or nothing would ever notice.
//
// Exported to the rest of the package's tests so they can say what they expect
// rather than only that something came back.
func fakeNextID(windowBytes []byte, length uint32) int32 {
	h := fnv.New32a()
	_, _ = h.Write(windowBytes)
	_, _ = h.Write(binary.LittleEndian.AppendUint32(nil, length))
	return int32(h.Sum32() % 120)
}

// fakeWindowBytes is fakeNextID's input, built from ids the way the frame does.
func fakeWindowBytes(window []int32) []byte {
	out := make([]byte, 0, len(window)*4)
	for _, id := range window {
		out = binary.LittleEndian.AppendUint32(out, uint32(id))
	}
	return out
}

// startFake writes the vocabulary a worker expects and returns the arguments
// NewWorker and NewPool need to spawn fakes in the given mode.
func startFake(t *testing.T, mode string) (exe, prefix string) {
	t.Helper()

	dir := t.TempDir()
	prefix = filepath.Join(dir, "fake")

	// As many symbols as the real checkpoint carries, as distinct ascending
	// bytes. The value only has to be self-consistent: the fake writes this file
	// and then reads its own ids back.
	alphabet := make([]byte, 120)
	for i := range alphabet {
		alphabet[i] = byte(i + 8)
	}
	if err := os.WriteFile(prefix+".vocab", alphabet, 0o600); err != nil {
		t.Fatalf("writing the fake vocabulary: %v", err)
	}

	t.Setenv(fakeWorkerEnv, mode) // restored when the test ends
	return os.Args[0], prefix
}

// oneWindow is a window of the width the protocol demands, with a recognisable
// tail so a test failure says which window it was.
// oneWindow is a window holding exactly these ids, padded to width on the right
// the way the scheduler pads. Its length -- what a caller must pass alongside it
// -- is len(tail).
func oneWindow(tail ...int32) []int32 {
	w := make([]int32, workerSeqLen)
	copy(w, tail)
	return w
}
