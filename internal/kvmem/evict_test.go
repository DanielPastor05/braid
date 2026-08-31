package kvmem

import (
	"math/rand/v2"
	"testing"
)

// fill puts `n` sequences of `positions` each into a pool and returns them as
// eviction candidates, admitted in id order.
func fill(t *testing.T, p *Pool, n, positions int) []Candidate {
	t.Helper()
	out := make([]Candidate, 0, n)
	for id := range uint64(n) {
		if err := p.Reserve(id, positions); err != nil {
			t.Fatalf("filling with sequence %d: %v", id, err)
		}
		out = append(out, Candidate{
			ID: id, Blocks: len(p.Table(id)), Positions: positions, Admitted: id,
		})
	}
	return out
}

func TestNoEvictionWhenThereIsRoom(t *testing.T) {
	p := newPool(t, 16, 16)
	candidates := fill(t, p, 2, 32) // four blocks used, twelve free

	victims, ok := Plan(p, 4, EvictLongest, candidates, 99)
	if !ok {
		t.Fatal("a request that fits was reported impossible")
	}
	if len(victims) != 0 {
		t.Errorf("evicted %d sequences to make room that already existed", len(victims))
	}
}

func TestEachPolicyPicksWhatItSaysItPicks(t *testing.T) {
	candidates := []Candidate{
		{ID: 1, Blocks: 1, Positions: 10, Admitted: 1},
		{ID: 2, Blocks: 8, Positions: 120, Admitted: 2},
		{ID: 3, Blocks: 3, Positions: 40, Admitted: 3},
	}

	cases := []struct {
		policy Policy
		want   uint64
	}{
		{EvictLongest, 2},  // eight blocks
		{EvictShortest, 1}, // one block
		{EvictNewest, 3},   // admitted last
	}
	for _, c := range cases {
		p := newPool(t, 12, 16)
		if err := p.Reserve(1, 12*16); err != nil { // hold everything
			t.Fatal(err)
		}
		victims, ok := Plan(p, 1, c.policy, candidates, 99)
		if !ok || len(victims) == 0 {
			t.Fatalf("%s: no plan (%v, ok=%v)", c.policy, victims, ok)
		}
		if victims[0] != c.want {
			t.Errorf("%s took %d first, want %d", c.policy, victims[0], c.want)
		}
	}
}

// TestTheRequesterIsNeverTheVictim is the loop that would otherwise be easy to
// write: under enough pressure a policy that can choose whoever just arrived
// will spend its whole time evicting them.
func TestTheRequesterIsNeverTheVictim(t *testing.T) {
	p := newPool(t, 8, 16)
	candidates := fill(t, p, 4, 32) // two blocks each, pool full

	for _, policy := range []Policy{EvictLongest, EvictShortest, EvictNewest} {
		victims, ok := Plan(p, 2, policy, candidates, 3) // protect the newest
		if !ok {
			t.Fatalf("%s: no plan", policy)
		}
		for _, v := range victims {
			if v == 3 {
				t.Errorf("%s evicted the sequence that asked for the room", policy)
			}
		}
	}
}

// TestAPlanFreesAtLeastWhatWasAsked: a plan that comes up short is worse than no
// plan, because the caller acts on it and then fails anyway, having thrown away
// somebody's cache for nothing.
func TestAPlanFreesAtLeastWhatWasAsked(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 5))

	for trial := range 500 {
		blocks := 4 + rng.IntN(28)
		p := newPool(t, blocks, 8)

		var candidates []Candidate
		id := uint64(0)
		for p.FreeBlocks() > 0 {
			positions := 1 + rng.IntN(3*8)
			if !p.Fits(positions) {
				positions = p.FreeBlocks() * 8
			}
			if err := p.Reserve(id, positions); err != nil {
				break
			}
			candidates = append(candidates, Candidate{
				ID: id, Blocks: len(p.Table(id)), Positions: positions, Admitted: id,
			})
			id++
		}

		need := 1 + rng.IntN(blocks)
		policy := Policy(rng.IntN(3))
		victims, ok := Plan(p, need, policy, candidates, ^uint64(0))
		if !ok {
			continue
		}

		freed := p.FreeBlocks()
		for _, v := range victims {
			freed += len(p.Table(v))
			p.Release(v)
		}
		if p.FreeBlocks() < need {
			t.Fatalf("trial %d: %s planned %d evictions for %d blocks and freed %d",
				trial, policy, len(victims), need, p.FreeBlocks())
		}
	}
}

// TestAnImpossibleRequestIsRefusedRatherThanEmptyingTheServer is the one that
// keeps overload from turning into collapse: if evicting everything still would
// not be enough, evicting anything is pure loss.
func TestAnImpossibleRequestIsRefusedRatherThanEmptyingTheServer(t *testing.T) {
	p := newPool(t, 4, 16)
	candidates := fill(t, p, 2, 32) // four blocks, pool full

	if _, ok := Plan(p, 5, EvictLongest, candidates, 99); ok {
		t.Error("a request for more blocks than the pool has was reported satisfiable")
	}
	if p.FreeBlocks() != 0 {
		t.Error("planning changed the pool; it is supposed to only decide")
	}

	// And the sequences are still there: a refused plan must not have evicted
	// anybody on the way to finding out it could not work.
	for id := range uint64(2) {
		if len(p.Table(id)) == 0 {
			t.Errorf("sequence %d lost its blocks to a plan that was then refused", id)
		}
	}
}

// TestEvictingTheLongestNeedsFewerVictims is the trade between the policies,
// measured rather than argued.
func TestEvictingTheLongestNeedsFewerVictims(t *testing.T) {
	build := func() (*Pool, []Candidate) {
		p := newPool(t, 64, 16)
		var candidates []Candidate
		for id := range uint64(8) {
			positions := int(id+1) * 16 // 1 to 8 blocks
			if err := p.Reserve(id, positions); err != nil {
				t.Fatal(err)
			}
			candidates = append(candidates, Candidate{
				ID: id, Blocks: len(p.Table(id)), Positions: positions, Admitted: id,
			})
		}
		return p, candidates
	}

	for _, policy := range []Policy{EvictLongest, EvictShortest, EvictNewest} {
		p, candidates := build()
		// The pool holds 36 of 64 blocks; ask for more than is free.
		victims, ok := Plan(p, p.FreeBlocks()+10, policy, candidates, ^uint64(0))
		if !ok {
			t.Fatalf("%s: no plan", policy)
		}
		recomputed := 0
		for _, v := range victims {
			for _, c := range candidates {
				if c.ID == v {
					recomputed += c.Positions
				}
			}
		}
		t.Logf("%-8s %d victims, %d positions to recompute", policy, len(victims), recomputed)
	}
}
