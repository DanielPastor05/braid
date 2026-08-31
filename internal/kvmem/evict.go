package kvmem

import "sort"

// Eviction: who loses their cache when there is not enough of it.
//
// The scheduler already has one way of saying no -- a full queue is a 429 at
// admission, which costs nobody anything. Eviction is the other one, and it is
// worse in every way: the victim has already been admitted, has already had
// forward passes spent on it, and is holding tokens a client has already been
// sent. Everything here is therefore about choosing the cheapest wrong answer.
//
// What it is *not* is a failure. A sequence whose cache is evicted is not
// cancelled: the scheduler still holds its history, so it can be readmitted and
// its cache rebuilt with a prefill. That is the same property that makes worker
// failover free -- the cache is never authoritative -- and it is why evicting is
// possible at all rather than being a fancy word for dropping a request.

// Policy decides which sequence gives up its blocks.
type Policy int

const (
	// EvictLongest takes from whoever holds the most, so one eviction is most
	// likely to be enough. It is also the most expensive single victim: the
	// longest sequence has the most to recompute on the way back.
	EvictLongest Policy = iota

	// EvictNewest takes from whoever was admitted last. It protects the work
	// already done -- a sequence two hundred tokens in has more sunk cost than
	// one that has produced three -- and it is the policy that makes progress
	// under sustained overload, because the alternative is a queue where nobody
	// ever finishes and everybody keeps being restarted.
	EvictNewest

	// EvictShortest takes from whoever holds the least, so the eviction costs
	// the least to undo. It is the one that can thrash: freeing a single block
	// at a time when a large reservation is waiting means evicting many
	// sequences to serve one.
	EvictShortest
)

func (p Policy) String() string {
	switch p {
	case EvictLongest:
		return "longest"
	case EvictNewest:
		return "newest"
	case EvictShortest:
		return "shortest"
	default:
		return "unknown"
	}
}

// Candidate is a sequence that could be evicted, and what it would cost.
type Candidate struct {
	ID uint64
	// Blocks it currently holds: what evicting it frees.
	Blocks int
	// Positions it has: what a prefill would have to recompute to bring it back.
	Positions int
	// Admitted orders candidates by age. Any monotone counter does; the
	// scheduler's step number is the obvious one.
	Admitted uint64
}

// Plan chooses victims until `blocks` are free, or reports that it cannot.
//
// It returns the whole set rather than one at a time so the caller can decide
// against the total: freeing nine blocks by evicting four sequences may be worse
// than refusing the request that wanted them, and only the caller knows which.
//
// `protect` is never evicted. The sequence that triggered the eviction belongs
// there: a policy that can choose the requester will, under enough pressure,
// spend its whole time evicting whoever just arrived.
func Plan(pool *Pool, need int, policy Policy, candidates []Candidate, protect uint64) ([]uint64, bool) {
	if need <= pool.FreeBlocks() {
		return nil, true
	}

	eligible := make([]Candidate, 0, len(candidates))
	for _, c := range candidates {
		if c.ID != protect && c.Blocks > 0 {
			eligible = append(eligible, c)
		}
	}

	// Sorted by the policy, then by id, so the same pressure produces the same
	// victims twice running. A tie broken by map order is a scheduler whose
	// behaviour under load cannot be reproduced.
	sort.Slice(eligible, func(i, j int) bool {
		a, b := eligible[i], eligible[j]
		switch policy {
		case EvictLongest:
			if a.Blocks != b.Blocks {
				return a.Blocks > b.Blocks
			}
		case EvictShortest:
			if a.Blocks != b.Blocks {
				return a.Blocks < b.Blocks
			}
		case EvictNewest:
			if a.Admitted != b.Admitted {
				return a.Admitted > b.Admitted
			}
		}
		return a.ID < b.ID
	})

	free := pool.FreeBlocks()
	victims := make([]uint64, 0, 4)
	for _, c := range eligible {
		if free >= need {
			break
		}
		victims = append(victims, c.ID)
		free += c.Blocks
	}

	if free < need {
		// Everything evictable is not enough. The caller must refuse rather
		// than evict the whole server and still fail.
		return nil, false
	}
	return victims, true
}

// Stats is what the eviction path has done, for /stats and for the curve the
// whole memory argument is supposed to produce.
type Stats struct {
	Admitted int64 `json:"admitted"`
	Refused  int64 `json:"refused"`
	Evicted  int64 `json:"evicted"`
	// Recomputed is the positions prefills have had to redo because of an
	// eviction. It is the real cost, and it is not the eviction count: evicting
	// one sequence of a thousand positions costs more than evicting ten of ten.
	Recomputed int64 `json:"recomputed_positions"`
}
