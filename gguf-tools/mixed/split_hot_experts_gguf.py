#!/usr/bin/env python3
"""Add per-layer hot-expert overlay slabs to a DeepSeek V4 Flash GGUF.

Overlay design (no permutation): the base 256-expert tensors stay untouched;
for each selected layer this tool appends four tensors:

  blk.N.ffn_gate_exps_hot.weight   [in, mid, H]   hot-quant
  blk.N.ffn_up_exps_hot.weight     [in, mid, H]   hot-quant
  blk.N.ffn_down_exps_hot.weight   [mid, out, H]  hot-quant
  blk.N.ffn_hot_ids                [H]            I32 global expert ids

Row h of a hot slab is the donor's (or base's) expert ffn_hot_ids[h]. The
engine dispatches hot experts from the slab and masks them out of the cold
pass; expert ids never change meaning, so the router, tid2eid hash tables and
e-score biases are untouched.

--hot-source base byte-copies the hot rows from the base itself (same quant):
the resulting model must produce identical logits to the base, which is the
phase-1 validation harness for the engine's split dispatch.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from splice_mixed_expert_layers_gguf import (  # noqa: E402
    TensorInfo,
    copy_exact,
    pack_string,
    pad_to,
    parse_gguf,
    parse_layer_set,
    qtype_name,
    tensor_nbytes,
)

GGML_TYPE_I32 = 26


def emit_tensor_info(out, name: str, dims: tuple[int, ...],
                     ggml_type: int, rel_offset: int) -> None:
    out.write(pack_string(name))
    out.write(struct.pack("<I", len(dims)))
    for dim in dims:
        out.write(struct.pack("<Q", dim))
    out.write(struct.pack("<I", ggml_type))
    out.write(struct.pack("<Q", rel_offset))


def pad_stream_to(out, target: int) -> None:
    gap = target - out.tell()
    assert gap >= 0
    if gap:
        out.write(b"\0" * gap)


def load_hot_lists(path: Path, hot_count: int) -> dict[int, list[int]]:
    """Top-`hot_count` expert ids per layer from the REAP aggregation TSV
    (layer\trank\texpert\tshare, rank 1-based already sorted)."""
    hot: dict[int, list[int]] = {}
    with path.open() as f:
        header = f.readline()
        if not header.startswith("layer"):
            raise ValueError(f"{path} does not look like a hot-expert TSV")
        for line in f:
            layer_s, rank_s, expert_s, _share = line.rstrip("\n").split("\t")
            layer, rank, expert = int(layer_s), int(rank_s), int(expert_s)
            if rank <= hot_count:
                hot.setdefault(layer, []).append(expert)
    for layer, ids in hot.items():
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate hot expert in layer {layer}")
        hot[layer] = sorted(ids)
    return hot


def expert_slab(t: TensorInfo, n_experts: int) -> int:
    if t.n_bytes % n_experts != 0:
        raise ValueError(f"{t.name}: {t.n_bytes} bytes not divisible by {n_experts} experts")
    return t.n_bytes // n_experts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--donor", type=Path, required=True,
                    help="source of hot rows (ignored with --hot-source base)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hot-list", type=Path, required=True,
                    help="TSV layer/rank/expert/share (REAP aggregation)")
    ap.add_argument("--hot-count", type=int, default=8)
    ap.add_argument("--layers", type=str, required=True,
                    help="layers to split, e.g. 3-36 or 0,3-36")
    ap.add_argument("--hot-source", choices=("donor", "base"), default="donor")
    ap.add_argument("--n-experts", type=int, default=256)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} exists (use --force)")
    layers = parse_layer_set(args.layers)
    hot_lists = load_hot_lists(args.hot_list, args.hot_count)

    base = parse_gguf(args.base)
    donor = base if args.hot_source == "base" else parse_gguf(args.donor)

    # Plan: every base tensor verbatim, then the new hot tensors.
    new_tensors: list[tuple[str, tuple[int, ...], int, TensorInfo, list[int]]] = []
    for layer in sorted(layers):
        ids = hot_lists.get(layer)
        if not ids or len(ids) != args.hot_count:
            raise SystemExit(f"layer {layer}: hot list has {len(ids or [])} ids, want {args.hot_count}")
        for proj in ("gate", "up", "down"):
            name = f"blk.{layer}.ffn_{proj}_exps.weight"
            src = donor.tensor_by_name.get(name)
            if src is None:
                raise SystemExit(f"{name} missing from {'base' if donor is base else 'donor'}")
            if src.dims[-1] != args.n_experts:
                raise SystemExit(f"{name}: last dim {src.dims[-1]} != {args.n_experts}")
            hot_dims = src.dims[:-1] + (args.hot_count,)
            new_tensors.append((f"blk.{layer}.ffn_{proj}_exps_hot.weight",
                                hot_dims, src.ggml_type, src, ids))
        # ids tensor sourced from nothing; encode inline
        new_tensors.append((f"blk.{layer}.ffn_hot_ids",
                            (args.hot_count,), GGML_TYPE_I32, None, ids))

    align = base.alignment
    rel = 0
    plans = []  # (name, dims, type, rel_offset, src_info_or_None, ids)
    for t in base.tensors:
        rel = pad_to(rel, align)
        plans.append((t.name, t.dims, t.ggml_type, rel, t, None))
        rel += t.n_bytes
    for name, dims, gtype, src, ids in new_tensors:
        rel = pad_to(rel, align)
        plans.append((name, dims, gtype, rel, src, ids))
        rel += tensor_nbytes(dims, gtype)

    hdr = struct.pack("<4sIQQ", b"GGUF", base.version,
                      len(plans), base.kv_count)
    with args.out.open("wb") as out, args.base.open("rb") as fbase:
        out.write(hdr)
        out.write(base.kv_blob)
        for name, dims, gtype, off, _src, _ids in plans:
            emit_tensor_info(out, name, dims, gtype, off)
        pad_stream_to(out, pad_to(out.tell(), align))
        data_start = out.tell()

        donor_f = fbase if donor is base else args.donor.open("rb")
        try:
            for name, dims, gtype, off, src, ids in plans:
                pad_stream_to(out, data_start + off)
                if src is not None and ids is None:
                    # verbatim base tensor
                    fbase.seek(src.data_offset)
                    copy_exact(fbase, out, src.n_bytes)
                elif src is not None:
                    # hot slab: gather expert rows from src (donor or base)
                    slab = expert_slab(src, args.n_experts)
                    for e in ids:
                        donor_f.seek(src.data_offset + e * slab)
                        copy_exact(donor_f, out, slab)
                else:
                    out.write(struct.pack(f"<{len(ids)}i", *ids))
        finally:
            if donor_f is not fbase:
                donor_f.close()

    added = len(new_tensors)
    print(f"wrote {args.out}: {len(plans)} tensors ({added} hot-overlay), "
          f"H={args.hot_count}, layers={sorted(layers)}, hot source={args.hot_source}")
    for layer in sorted(layers):
        gate = f"blk.{layer}.ffn_gate_exps_hot.weight"
        info = next(p for p in plans if p[0] == gate)
        print(f"  blk.{layer}: hot ids {hot_lists[layer]} ({qtype_name(info[2])})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
