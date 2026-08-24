package backend

import "errors"

var (
	// errRagged is returned when the three per-sequence slices handed to Step
	// disagree in length, which means the scheduler built the batch wrong.
	errRagged = errors.New("backend: windows, temperatures and seeds differ in length")

	// errWindowWidth is returned when a window is not exactly SeqLen ids. The
	// model has a fixed context and cannot be handed a short row.
	errWindowWidth = errors.New("backend: window is not SeqLen ids wide")
)
