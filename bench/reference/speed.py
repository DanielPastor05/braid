"""How fast is the engine's forward against PyTorch's, on the same card?

Same weights, same shapes, same GPU, one process at a time. This is deliberately
*not* a server-against-server comparison: braid's scheduler and PyTorch's absence
of one would be most of the difference, and the question here is about the
arithmetic underneath. `cpp-ai-engine` publishes "1.70x slower than PyTorch" for
training; this is the serving half of the same question.

The measurement points are the ones braid already reports, so the numbers line up
with the tables in the README: batches of 1, 8 and 32, at widths from one
position to the full 256 context. Width matters as much as batch here — braid
runs a step at the width of the longest sequence in it, whose mean under load is
29, so the interesting rows are the narrow ones and not the full-context one.

    python bench/reference/speed.py models/charlm --repeats 100

Run `parity.py` first. A speed number for an implementation that computes
something else is worse than no number at all, because it looks like a result.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import charlm  # noqa: E402

WIDTHS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
BATCHES = (1, 8, 32)


def time_forward(model: charlm.CharModel, ids: torch.Tensor, mask: torch.Tensor,
                 repeats: int, device: torch.device) -> float:
    """Milliseconds per forward, synchronised.

    CUDA kernels are asynchronous, so a loop that only launches them measures how
    fast Python can launch. The synchronise inside the loop is what the engine's
    own benchmark does, and matching the method matters more than the method
    being the cheapest available.
    """
    with torch.no_grad():
        for _ in range(3):  # warm: allocator, autotuner, lazy module init
            model(ids, mask)
        if device.type == "cuda":
            torch.cuda.synchronize()

        started = time.perf_counter()
        for _ in range(repeats):
            model(ids, mask)
            if device.type == "cuda":
                torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    return elapsed * 1000.0 / repeats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", nargs="?", default="models/charlm")
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("no CUDA device: this would compare a CPU against a GPU", file=sys.stderr)
        return 2

    # Pinned, not assumed. On Ampere, TF32 makes a matmul faster by making it
    # less precise -- 10 mantissa bits against 23 -- and the engine has no such
    # path, so a PyTorch that quietly enabled it would win this comparison by
    # computing something cheaper rather than by computing it better. The
    # defaults are already what is wanted in torch 2.6; they were not always, and
    # a benchmark whose fairness depends on a default is a benchmark waiting to
    # become wrong.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    model, alphabet = charlm.load(args.prefix, device)
    vocab = len(alphabet)
    print(f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
          f"{args.repeats} repeats, tf32 {torch.backends.cuda.matmul.allow_tf32}, "
          f"fp32 precision {torch.get_float32_matmul_precision()}", file=sys.stderr)

    print("| batch | window | torch ms |")
    print("|---|---|---|")
    for batch in BATCHES:
        for width in WIDTHS:
            ids = torch.arange(batch * width, device=device).remainder(vocab).view(batch, width)
            mask = charlm.causal_mask(width, device)
            ms = time_forward(model, ids, mask, args.repeats, device)
            print(f"| {batch} | {width} | {ms:.3f} |")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
