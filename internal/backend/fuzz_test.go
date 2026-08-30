package backend

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"io"
	"testing"
)

// Fuzzing the half of the protocol this side does not control.
//
// braid writes the request frame and can be trusted to write it correctly; it
// *reads* the response, and everything in that response arrives from another
// process. The worker is a child this server spawned, so this is not a
// network-facing parser -- but a worker that is confused is enough, and the
// engine's own fuzz target for its weight parser exists for the same reason.
//
// The properties are the two that matter for a server: it must not panic, and it
// must not believe a length. The first version of exchange() read a uint32 and
// allocated it, so four bytes at the wrong offset were a four-gigabyte
// allocation and the process died of an error message.

type discardWriteCloser struct{ io.Writer }

func (discardWriteCloser) Close() error { return nil }

// readerWorker is a Worker whose pipe is a fixed slice of bytes. Nothing about
// exchange() needs a process, which is what makes it fuzzable at all.
func readerWorker(response []byte) *Worker {
	return &Worker{
		stdin:  discardWriteCloser{io.Discard},
		stdout: bufio.NewReader(bytes.NewReader(response)),
		seqLen: workerSeqLen,
	}
}

func okResponse(n int) []byte {
	out := binary.LittleEndian.AppendUint32(nil, statusOK)
	for i := range n {
		out = binary.LittleEndian.AppendUint32(out, uint32(i))
	}
	for range 7 {
		out = binary.LittleEndian.AppendUint64(out, 1)
	}
	return out
}

func errorResponse(message string) []byte {
	out := binary.LittleEndian.AppendUint32(nil, statusError)
	out = binary.LittleEndian.AppendUint32(out, uint32(len(message)))
	return append(out, message...)
}

func FuzzWorkerReadsAResponse(f *testing.F) {
	f.Add(okResponse(1), uint8(1))
	f.Add(okResponse(4), uint8(4))
	f.Add(errorResponse("out of memory"), uint8(1))
	f.Add(errorResponse(""), uint8(1))
	f.Add([]byte{}, uint8(1))                                  // the worker died before answering
	f.Add(okResponse(4)[:6], uint8(4))                         // truncated mid-id
	f.Add(binary.LittleEndian.AppendUint32(nil, 99), uint8(1)) // a status that is not one

	// The one that mattered: an error frame announcing four gigabytes of text.
	huge := binary.LittleEndian.AppendUint32(nil, statusError)
	huge = binary.LittleEndian.AppendUint32(huge, ^uint32(0))
	f.Add(huge, uint8(1))

	f.Fuzz(func(t *testing.T, response []byte, rows uint8) {
		n := int(rows)%16 + 1

		ids, timings, err := readerWorker(response).exchange(nil, n)

		if err != nil {
			// A refusal is always allowed: most inputs are not valid frames.
			// What is not allowed is a refusal that also returns something.
			if ids != nil {
				t.Errorf("exchange returned %d ids alongside an error", len(ids))
			}
			return
		}

		// Having accepted the frame, the shape it promised has to be the shape
		// it gives. A parser that returns fewer ids than the caller asked for
		// makes the scheduler index past the end of its own batch.
		if len(ids) != n {
			t.Errorf("exchange accepted a frame and returned %d ids for %d rows", len(ids), n)
		}
		_ = timings
	})
}

// TestAnErrorMessageLongerThanTheCapIsRefused is the fuzz finding, pinned as a
// test so it cannot come back quietly.
//
// The length arrives from the worker and the server allocates it. Before the
// cap, a worker that wrote a wrong length -- confused, not malicious -- asked
// this process for four gigabytes.
func TestAnErrorMessageLongerThanTheCapIsRefused(t *testing.T) {
	frame := binary.LittleEndian.AppendUint32(nil, statusError)
	frame = binary.LittleEndian.AppendUint32(frame, ^uint32(0))

	_, _, err := readerWorker(frame).exchange(nil, 1)
	if err == nil {
		t.Fatal("a four-gigabyte error message was accepted")
	}
	if !bytes.Contains([]byte(err.Error()), []byte("which is not a message")) {
		t.Errorf("the refusal does not say what was wrong: %v", err)
	}

	// And a message at the cap still works, so the bound is a bound and not a
	// ban on error messages.
	long := make([]byte, maxErrorMessage)
	for i := range long {
		long[i] = 'x'
	}
	_, _, err = readerWorker(errorResponse(string(long))).exchange(nil, 1)
	if err == nil {
		t.Fatal("a legitimate error frame at the cap was silently accepted as success")
	}
	if !bytes.Contains([]byte(err.Error()), []byte("refused the step")) {
		t.Errorf("a message at the cap was not surfaced as the worker's refusal: %v", err)
	}
}
