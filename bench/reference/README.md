# The other implementation

`cpp-ai-engine` opens with "1.70× slower than PyTorch, and that is the number
worth publishing". This is the serving half of the same question: the same model,
the same weights, the same card, one process at a time.

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

```bash
python bench/reference/parity.py models/charlm --trials 200
```

```bash
python bench/reference/speed.py models/charlm --repeats 100
```

Run them in that order. `parity.py` is not a formality — it is what makes the
second number mean anything, and the ways to get a transcription subtly wrong are
ordinary ones: pre-norm written as post-norm, a Linear loaded without its
transpose (which on a square matrix loads cleanly and computes nonsense), the
positional encoding's pair index off by an integer division. All three produce
plausible logits.

The engine side of the comparison is `braid_bench_decode`, in the repository
root's `engine/`, run at the same batch sizes and widths.

| file | |
|---|---|
| `charlm.py` | the model, and a reader for the engine's checkpoint format |
| `parity.py` | do the two pick the same token? Compared against `braid_worker` over its own protocol, so it is the thing braid actually serves with |
| `speed.py` | how long a forward takes, at the points braid reports |

`speed.py` pins `allow_tf32 = False` and `float32_matmul_precision = "highest"`
rather than trusting the defaults. TF32 makes an Ampere matmul faster by making
it less precise — 10 mantissa bits against 23 — and the engine has no such path,
so a PyTorch that quietly enabled it would win by computing something cheaper
instead of by computing it better.
