package sched

import (
	"context"
	"testing"
)

// A regression gate that measures allocations rather than time.
//
// A timing baseline is the obvious thing and it is the wrong thing: CI runners
// vary by more than any regression worth catching, so the threshold has to be
// loose enough to be useless or tight enough to be flaky. Allocation counts have
// neither problem. They are deterministic, they are identical on every machine,
// and they are what actually regresses when somebody adds a per-token map lookup
// or forgets that a slice literal escapes.
//
// The numbers below were measured, not chosen, and the ceilings are the measured
// value with room for one honest change. When one of them fires, the question is
// which allocation appeared -- `go test -bench . -benchmem -memprofile` answers
// it in a minute.

func benchScheduler(b *testing.B, batch int) *Scheduler {
	b.Helper()

	mock := fastMock()
	s, err := New(mock, Config{MaxBatch: batch, QueueDepth: 512, MaxTokensLimit: 1024})
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { _ = s.Close() })
	return s
}

// drainOne runs a whole generation and returns nothing, so the benchmark
// measures the loop rather than the caller's bookkeeping.
func drainOne(b *testing.B, s *Scheduler, tokens int) {
	b.Helper()

	stream, done, err := s.Submit(context.Background(), Request{
		Prompt: "the engine ", MaxTokens: tokens, Temperature: 0.8, Seed: 1,
	})
	if err != nil {
		b.Fatal(err)
	}
	for range stream {
	}
	if res := <-done; res.Err != nil {
		b.Fatal(res.Err)
	}
}

// BenchmarkGeneration is one sequence from submission to its last token, which
// is the path every request takes.
func BenchmarkGeneration(b *testing.B) {
	s := benchScheduler(b, 32)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		drainOne(b, s, 32)
	}
}

// BenchmarkGenerationInABatch runs sixteen at once, so the per-step work is
// divided by sixteen and what is left is the per-sequence and per-token cost.
func BenchmarkGenerationInABatch(b *testing.B) {
	s := benchScheduler(b, 32)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		done := make([]<-chan Result, 16)
		for i := range done {
			stream, finished, err := s.Submit(context.Background(), Request{
				Prompt: "the engine ", MaxTokens: 32, Temperature: 0.8, Seed: uint64(i),
			})
			if err != nil {
				b.Fatal(err)
			}
			done[i] = finished
			go func() {
				for range stream {
				}
			}()
		}
		for _, finished := range done {
			if res := <-finished; res.Err != nil {
				b.Fatal(res.Err)
			}
		}
	}
}

// TestTheHotLoopDoesNotGrowItsAllocations is the gate.
//
// It runs the benchmark inside an ordinary test so CI catches a regression on
// every push rather than only when somebody remembers to run -bench.
func TestTheHotLoopDoesNotGrowItsAllocations(t *testing.T) {
	if testing.Short() {
		t.Skip("the gate runs a benchmark; -short skips it")
	}

	cases := []struct {
		name    string
		bench   func(*testing.B)
		tokens  int
		ceiling float64 // allocations per generated token
	}{
		// Measured at 2.25 and 1.38, three runs, identical to the hundredth --
		// which is the property that makes this a gate rather than a mood. The
		// ceilings are one allocation per token above each, so a single new
		// allocation in the loop fires them and ordinary noise cannot.
		{"one at a time", BenchmarkGeneration, 32, 3.0},
		{"sixteen at once", BenchmarkGenerationInABatch, 32 * 16, 2.2},
	}

	for _, c := range cases {
		result := testing.Benchmark(c.bench)
		if result.N == 0 {
			t.Fatalf("%s: the benchmark did not run", c.name)
		}
		perToken := float64(result.AllocsPerOp()) / float64(c.tokens)
		t.Logf("%-16s %6.2f allocations per token (%d per operation, ceiling %.1f)",
			c.name, perToken, result.AllocsPerOp(), c.ceiling)

		if perToken > c.ceiling {
			t.Errorf("%s: %.2f allocations per token, ceiling %.1f. "+
				"Something in the loop started allocating; "+
				"`go test -bench . -benchmem -memprofile mem.out ./internal/sched` says what",
				c.name, perToken, c.ceiling)
		}
	}
}

// The floor is not zero and cannot be: Decode returns a string per token and a
// string is an allocation, and every request brings a channel and a sequence.
// What this catches is the difference between a constant number of allocations
// per token and a number that grows with the batch or with the history, which is
// the shape every accidental O(n) in a scheduler takes.
//
// The batched figure being *lower* than the solo one is the loop working: the
// per-step allocations are divided by sixteen there, so what is left is closer
// to the true per-token cost.
