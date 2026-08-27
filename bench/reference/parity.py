"""Do the two implementations compute the same thing?

This runs before any timing, and nothing downstream is worth reading until it
passes. A reference implementation that is 3x faster and subtly different is not
a reference implementation, it is a second model — and the ways to get it subtly
different are not exotic. Pre-norm written as post-norm. A Linear loaded without
its transpose, which on a square matrix loads cleanly and computes nonsense. The
positional encoding's pair index off by an integer division. Every one of those
produces plausible logits.

So: the same ids, the same mask, the same weights, both sides, and compare.

The comparison is to a tolerance rather than exact, and that is not a concession.
Both run in float32 on the same card and neither promises the other's reduction
order; the device compiler fuses multiply and add where the other does not, which
rounds once where the other rounds twice. Demanding bit-identity would be
demanding that one of them compute worse.

What is checked exactly is the **argmax**: which token each position would pick.
A difference in the last bits that never changes the answer is arithmetic; one
that does is a bug wearing arithmetic's clothes.

    python bench/reference/parity.py models/charlm

The engine side comes from `braid_worker` over its own protocol, so this compares
against the thing braid actually serves with, not against a C++ program written
for the occasion.
"""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import charlm  # noqa: E402

FRAME_MAGIC = 0x36445242  # 'BRD6'
STATUS_OK = 0


class Worker:
    """braid_worker over BRD6, for one batch at a time.

    The protocol only returns sampled ids, not logits, which is the right thing
    for a server and an awkward thing for a parity check. It is also enough: at
    temperature near zero the sampler picks the argmax, so the id it returns is
    the argmax of the row it sampled. That is what this compares against.
    """

    def __init__(self, exe: str, prefix: str) -> None:
        # Resolved rather than passed through: Windows' CreateProcess does not
        # search the working directory the way a shell does, so a perfectly good
        # relative path fails with "the system cannot find the file specified".
        path = Path(exe)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"no worker at {path}; build it or pass --worker")
        self.proc = subprocess.Popen(
            [str(path), prefix], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL)

    def step(self, windows: torch.Tensor, lengths: list[int],
             temperature: float, seeds: list[int]) -> list[int]:
        n, width = windows.shape
        if width != charlm.SEQ_LEN:
            raise ValueError(f"the frame is {charlm.SEQ_LEN} ids wide, got {width}")
        frame = bytearray()
        frame += struct.pack("<II", FRAME_MAGIC, n)
        frame += windows.to(torch.int32).numpy().tobytes()
        frame += struct.pack(f"<{n}i", *lengths)
        frame += struct.pack(f"<{n}f", *([temperature] * n))
        frame += struct.pack(f"<{n}Q", *seeds)
        self.proc.stdin.write(bytes(frame))
        self.proc.stdin.flush()

        (status,) = struct.unpack("<I", self._read(4))
        if status != STATUS_OK:
            (length,) = struct.unpack("<I", self._read(4))
            raise RuntimeError(f"the worker refused: {self._read(length).decode()}")
        ids = struct.unpack(f"<{n}i", self._read(4 * n))
        self._read(7 * 8)  # the timings, which this does not need
        return list(ids)

    def _read(self, n: int) -> bytes:
        out = self.proc.stdout.read(n)
        if out is None or len(out) != n:
            raise RuntimeError("the worker closed the pipe")
        return out

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait(timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", nargs="?", default="models/charlm")
    parser.add_argument("--worker", default="build/braid_worker.exe")
    parser.add_argument("--trials", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet = charlm.load(args.prefix, device)
    vocab = len(alphabet)
    print(f"torch {torch.__version__} on {device}, "
          f"{sum(p.numel() for p in model.parameters()):,} parameters, {vocab} symbols")

    generator = torch.Generator().manual_seed(20260827)
    worker = Worker(args.worker, args.prefix)

    mismatches = 0
    rows = 0

    try:
        for trial in range(args.trials):
            # Batches of mixed length, because that is what the server sees and
            # because a row's width depends on its neighbours: braid runs the
            # batch at the width of its longest row.
            n = 1 + (trial % 4)
            lengths = [1 + int(torch.randint(charlm.SEQ_LEN, (1,), generator=generator))
                       for _ in range(n)]
            width = max(lengths)

            windows = torch.zeros(n, charlm.SEQ_LEN, dtype=torch.int64)
            for i, length in enumerate(lengths):
                windows[i, :length] = torch.randint(vocab, (length,), generator=generator)

            with torch.no_grad():
                logits = model(windows[:, :width].to(device),
                               charlm.causal_mask(width, device))

            # A temperature low enough that the inverse-CDF sampler lands on the
            # argmax, whatever seed it is given.
            got = worker.step(windows, lengths, 1e-4, [7] * n)

            for i, length in enumerate(lengths):
                row = logits[i, length - 1]
                torch_id = int(row.argmax())
                if torch_id != got[i]:
                    if mismatches < 5:
                        top = row.topk(2)
                        # The gap between the top two logits says which kind of
                        # disagreement this is. A tiny gap is two implementations
                        # rounding differently on a coin flip; a wide one is a bug.
                        print(f"  trial {trial} row {i} (length {length}, width {width}): "
                              f"torch {torch_id} vs engine {got[i]}; "
                              f"top two logits differ by {float(top.values[0] - top.values[1]):.2e}")
                    mismatches += 1
                rows += 1
    finally:
        worker.close()

    print(f"{rows} rows compared over {args.trials} batches")
    print(f"argmax disagreements: {mismatches}")

    if mismatches:
        print("\nFAIL: the two implementations pick different tokens.")
        return 1
    print("\nThe two implementations pick the same token on every row.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
