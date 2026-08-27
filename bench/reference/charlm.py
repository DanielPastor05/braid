"""The same model, in PyTorch, reading the same weights.

This repository has never compared itself to anything. `cpp-ai-engine` opens with
"1.70x slower than PyTorch, and that is the number worth publishing"; braid opens
with numbers that are only meaningful against themselves. This is the other half.

The model is the one in `engine/charmodel.hpp`, transcribed rather than ported:
6 pre-norm blocks, 384 wide, 6 heads, 1536 feed-forward, 256-id context, a
145-symbol byte alphabet, sinusoidal positions. 10 758 289 parameters, and the
count is asserted rather than assumed.

**The weights are the engine's own**, read straight out of `models/charlm.bin` --
a format simple enough to parse in thirty lines of `struct`, which is the only
reason this comparison is honest. A reimplementation trained separately would
compare two models. This compares two implementations of one.

Parity is checked before speed is, because a fast wrong answer is not a
comparison. See `parity.py`.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Geometry, from engine/charmodel.hpp. Kept as constants rather than inferred
# from the checkpoint so that a checkpoint from a different model is rejected by
# the shape check rather than quietly reshaping this one.
SEQ_LEN = 256
D_MODEL = 384
HEADS = 6
FEED_FORWARD = 1536
BLOCKS = 6
HEAD_DIM = D_MODEL // HEADS
LAYER_NORM_EPS = 1e-5  # the engine's default, and PyTorch's


def read_checkpoint(path: str | Path) -> dict[str, torch.Tensor]:
    """Reads the engine's own weight format.

    "CPPAIENG", version, n_tensors, then per tensor: name length, name, ndim,
    dimensions as u64, and the float32 data. Little-endian, self-describing,
    matched by name.
    """
    out: dict[str, torch.Tensor] = {}
    with open(path, "rb") as f:
        signature = f.read(8)
        if signature != b"CPPAIENG":
            raise ValueError(f"{path} is not an engine checkpoint: {signature!r}")
        version, count = struct.unpack("<II", f.read(8))
        if version != 1:
            raise ValueError(f"{path} is version {version}, this reader knows 1")
        for _ in range(count):
            (name_len,) = struct.unpack("<I", f.read(4))
            name = f.read(name_len).decode()
            (ndim,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack(f"<{ndim}Q", f.read(8 * ndim))
            n = math.prod(dims)
            raw = f.read(4 * n)
            if len(raw) != 4 * n:
                raise ValueError(f"{path} is truncated inside {name}")
            out[name] = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(dims)
    return out


def positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """The engine's `positional_encoding`, to the letter.

    Transcribed from src/transformer.cpp rather than taken from any of the
    several subtly different versions in circulation: the pair index is `i // 2`
    with integer division, so dimensions 0 and 1 share a frequency, 2 and 3 share
    the next, and so on. Getting that wrong gives a model that still runs and
    quietly attends to the wrong places.
    """
    i = torch.arange(d_model, dtype=torch.float32)
    pair = torch.div(i, 2, rounding_mode="floor")
    exponent = 2.0 * pair / d_model
    pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    angle = pos / torch.pow(torch.tensor(10000.0), exponent).unsqueeze(0)
    pe = torch.where(i % 2 == 0, torch.sin(angle), torch.cos(angle))
    return pe


class Block(nn.Module):
    """One pre-norm transformer block, matching `engine::nn::TransformerBlock`.

    Pre-norm: the residual path carries the unnormalised input, so
    `h = x + attn(norm1(x))` and `out = h + ff2(relu(ff1(norm2(h))))`. Getting
    the order wrong trains and infers perfectly well and produces different
    numbers, which is exactly the class of difference the parity check exists
    to catch.
    """

    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(D_MODEL, eps=LAYER_NORM_EPS)
        self.norm2 = nn.LayerNorm(D_MODEL, eps=LAYER_NORM_EPS)
        self.query = nn.Linear(D_MODEL, D_MODEL)
        self.key = nn.Linear(D_MODEL, D_MODEL)
        self.value = nn.Linear(D_MODEL, D_MODEL)
        self.out = nn.Linear(D_MODEL, D_MODEL)
        self.ff1 = nn.Linear(D_MODEL, FEED_FORWARD)
        self.ff2 = nn.Linear(FEED_FORWARD, D_MODEL)

    def attention(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape

        def heads(projected: torch.Tensor) -> torch.Tensor:
            return projected.view(batch, seq, HEADS, HEAD_DIM).permute(0, 2, 1, 3)

        q, k, v = heads(self.query(x)), heads(self.key(x)), heads(self.value(x))
        scores = q @ k.transpose(-2, -1) / math.sqrt(HEAD_DIM)
        weights = torch.softmax(scores + mask, dim=-1)
        merged = (weights @ v).permute(0, 2, 1, 3).reshape(batch, seq, D_MODEL)
        return self.out(merged)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = x + self.attention(self.norm1(x), mask)
        return h + self.ff2(F.relu(self.ff1(self.norm2(h))))


class CharModel(nn.Module):
    def __init__(self, vocab: int) -> None:
        super().__init__()
        self.vocab = vocab
        self.embedding = nn.Embedding(vocab, D_MODEL)
        self.blocks = nn.ModuleList(Block() for _ in range(BLOCKS))
        self.head = nn.Linear(D_MODEL, vocab)
        self.register_buffer("positions", positional_encoding(SEQ_LEN, D_MODEL))

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(B, S) ids in, (B, S, vocab) logits out.

        S may be narrower than SEQ_LEN. braid's worker runs at the width of the
        longest sequence in the batch rather than the model's full context, so a
        comparison that always ran 256 positions would be comparing against
        something braid stopped doing.
        """
        take = ids.shape[1]
        h = self.embedding(ids) + self.positions[:take]
        for block in self.blocks:
            h = block(h, mask)
        return self.head(h)


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Additive mask: 0 where attending is allowed, -1e9 where it is not.

    The engine's value, not `-inf`. They are not interchangeable: -1e9 leaves a
    finite number in the softmax and -inf can produce a NaN in a row that is
    entirely masked. Matching it is part of matching the arithmetic.
    """
    mask = torch.zeros(seq_len, seq_len, device=device)
    mask.masked_fill_(torch.triu(torch.ones_like(mask), diagonal=1).bool(), -1e9)
    return mask


def load(prefix: str | Path, device: torch.device) -> tuple[CharModel, bytes]:
    """Builds the model and fills it from the engine's checkpoint.

    The name mangling is the engine's, and it is not pretty -- a Linear's
    parameters are named after its `name()`, so they arrive as
    `block3ff1.Linear(384 -> 1536, bias=true).0` for the weight and `.1` for the
    bias. Matching on the readable prefix rather than reproducing the whole
    string keeps this from breaking the day somebody edits a `name()`.

    Linear weights are stored (in, out) and PyTorch wants (out, in), so every one
    of them is transposed on the way in. A missed transpose on a square matrix --
    and every attention projection here is square -- loads without complaint and
    computes something plausible. That is what the parity check is for.
    """
    prefix = Path(prefix)
    alphabet = prefix.with_suffix(".vocab").read_bytes()
    tensors = read_checkpoint(prefix.with_suffix(".bin"))

    model = CharModel(len(alphabet))
    state: dict[str, torch.Tensor] = {"embedding.weight": _one(tensors, "embeddingembedding.weight")}

    for i in range(BLOCKS):
        block = f"block{i}"
        state[f"blocks.{i}.norm1.weight"] = _one(tensors, f"{block}norm1.layernorm.gamma")
        state[f"blocks.{i}.norm1.bias"] = _one(tensors, f"{block}norm1.layernorm.beta")
        state[f"blocks.{i}.norm2.weight"] = _one(tensors, f"{block}norm2.layernorm.gamma")
        state[f"blocks.{i}.norm2.bias"] = _one(tensors, f"{block}norm2.layernorm.beta")
        linears = {
            "query": f"{block}attn.query.Linear",
            "key": f"{block}attn.key.Linear",
            "value": f"{block}attn.value.Linear",
            "out": f"{block}attn.out.Linear",
            "ff1": f"{block}ff1.Linear",
            "ff2": f"{block}ff2.Linear",
        }
        for name, key in linears.items():
            state[f"blocks.{i}.{name}.weight"] = _weight(tensors, key)
            state[f"blocks.{i}.{name}.bias"] = _bias(tensors, key)

    state["head.weight"] = _weight(tensors, "headLinear")
    state["head.bias"] = _bias(tensors, "headLinear")

    # strict=False only because `positions` is a buffer this builds rather than
    # loads; everything else is then checked by hand, because a silently
    # unfilled parameter is a model full of random numbers that still runs.
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if k != "positions"]
    if missing:
        raise RuntimeError(f"the checkpoint did not fill: {missing}")
    if unexpected:
        raise RuntimeError(f"the checkpoint brought parameters with no slot: {unexpected}")

    total = sum(p.numel() for p in model.parameters())
    if total != 10_758_289:
        raise RuntimeError(f"{total} parameters, expected 10 758 289 -- the geometry has drifted")

    return model.to(device).eval(), alphabet


def _matching(tensors: dict[str, torch.Tensor], prefix_text: str, suffix: str) -> torch.Tensor:
    keys = [k for k in tensors if k.startswith(prefix_text) and k.endswith(suffix)]
    if len(keys) != 1:
        raise KeyError(f"{prefix_text!r}...{suffix!r} matched {len(keys)} tensors, wanted 1")
    return tensors[keys[0]]


def _one(tensors: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    """A parameter whose name the engine did not mangle: embeddings and the two
    LayerNorm vectors. Only Linear names carry the `.0`/`.1` suffixes."""
    if name not in tensors:
        raise KeyError(f"{name!r} is not in the checkpoint")
    return tensors[name]


def _weight(tensors: dict[str, torch.Tensor], prefix_text: str) -> torch.Tensor:
    """(in, out) in the file, (out, in) in PyTorch."""
    return _matching(tensors, prefix_text, ".0").T.contiguous()


def _bias(tensors: dict[str, torch.Tensor], prefix_text: str) -> torch.Tensor:
    """Stored (1, out); PyTorch wants (out,)."""
    return _matching(tensors, prefix_text, ".1").squeeze(0)
