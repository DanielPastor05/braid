package backend

import "errors"

var (
	// errRagged is returned when the per-sequence slices handed to Step
	// disagree in length, which means the scheduler built the batch wrong.
	errRagged = errors.New("backend: the per-sequence slices differ in length")

	// errWindowWidth is returned when a window is not exactly SeqLen ids. The
	// model has a fixed context and cannot be handed a short row.
	errWindowWidth = errors.New("backend: window is not SeqLen ids wide")

	// errWindowLength is returned when a row claims more real ids than its
	// window holds, or none at all. Both would sample a position that is not
	// the end of the sequence, which reads as a working server producing
	// nonsense rather than as a failure.
	errWindowLength = errors.New("backend: window length is outside 1..SeqLen")
)
