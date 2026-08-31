package backend

import (
	"context"
	"encoding/binary"
	"hash/fnv"
	"sync/atomic"
	"time"
)

// Mock is a backend with no model behind it, used by the scheduler's tests and
// by the load harness when no GPU is present.
//
// The next token is a hash of the sequence's real ids, its length, its
// temperature and its seed. That is not a language model, but it has the three
// properties the scheduler is tested against and a real model also has: the
// answer depends on every real id, so a scheduler that crosses two rows produces
// visibly different text; it depends on nothing else, so the same sequence must
// decode identically whoever it shares a batch with; and it does not depend on
// the padding, because with the padding on the right the model cannot reach it
// either. A mock that hashed the padding too would go on passing if the length
// were ever threaded through wrong.
type Mock struct {
	// Base is the cost of a step regardless of how many sequences are in it --
	// kernel launches, the host-device round trip, the parts a batch amortises.
	Base time.Duration
	// PerSeq is the marginal cost of one more sequence in the same step.
	PerSeq time.Duration

	seqLen int
	vocab  int

	steps    atomic.Int64
	sequence atomic.Int64 // sequences advanced, summed over steps
}

// NewMock returns a mock with the character model's geometry -- a 256-id window
// over a 145-symbol alphabet -- and a cost curve where a step is mostly fixed.
//
// The default timings are not measured from anything -- they are round numbers
// chosen so a load test against the mock produces the shape of the real curve
// rather than its values. Every number this project publishes comes from the
// CUDA backend; the mock exists to test logic, not to be quoted.
func NewMock() *Mock {
	return &Mock{
		Base:   8 * time.Millisecond,
		PerSeq: 200 * time.Microsecond,
		seqLen: workerSeqLen,
		vocab:  145,
	}
}

func (m *Mock) SeqLen() int    { return m.seqLen }
func (m *Mock) VocabSize() int { return m.vocab }
func (m *Mock) Close() error   { return nil }

// Steps is how many times Step has been called, and Sequences is the total
// number of sequences advanced across those calls. Their ratio is the mean
// batch size, which is the number the whole project is about.
func (m *Mock) Steps() int64     { return m.steps.Load() }
func (m *Mock) Sequences() int64 { return m.sequence.Load() }

func (m *Mock) Step(ctx context.Context, windows [][]int32, lengths []int32, slots []int32,
	temperatures []float32, seeds []uint64) ([]int32, error) {
	if len(windows) != len(temperatures) || len(windows) != len(seeds) ||
		len(windows) != len(lengths) {
		return nil, errRagged
	}
	for i, w := range windows {
		if len(w) != m.seqLen {
			return nil, errWindowWidth
		}
		if lengths[i] < 1 || int(lengths[i]) > m.seqLen {
			return nil, errWindowLength
		}
	}

	// Honoured rather than ignored, so that a test can cancel a slow step the
	// way a real one would be cancelled.
	if d := m.Base + time.Duration(len(windows))*m.PerSeq; d > 0 {
		timer := time.NewTimer(d)
		defer timer.Stop()
		select {
		case <-timer.C:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	out := make([]int32, len(windows))
	for i, w := range windows {
		h := fnv.New64a()
		var buf [8]byte
		for _, id := range w[:lengths[i]] {
			binary.LittleEndian.PutUint32(buf[:4], uint32(id))
			_, _ = h.Write(buf[:4])
		}
		binary.LittleEndian.PutUint32(buf[:4], uint32(lengths[i]))
		_, _ = h.Write(buf[:4])
		binary.LittleEndian.PutUint64(buf[:], seeds[i])
		_, _ = h.Write(buf[:])
		binary.LittleEndian.PutUint32(buf[:4], uint32(temperatures[i]*1000))
		_, _ = h.Write(buf[:4])
		out[i] = int32(h.Sum64() % uint64(m.vocab))
	}

	m.steps.Add(1)
	m.sequence.Add(int64(len(windows)))
	return out, nil
}

// Encode and Decode treat the vocabulary as the first VocabSize bytes, which is
// close enough to the engine's character alphabet for a test and needs no
// corpus to build.
func (m *Mock) Encode(text string) []int32 {
	ids := make([]int32, 0, len(text))
	for _, b := range []byte(text) {
		if int(b) < m.vocab {
			ids = append(ids, int32(b))
		}
	}
	return ids
}

func (m *Mock) Decode(ids []int32) string {
	out := make([]byte, len(ids))
	for i, id := range ids {
		out[i] = byte(id)
	}
	return string(out)
}
