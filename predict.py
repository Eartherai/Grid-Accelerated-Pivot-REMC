"""Fold a sequence, or score and replay a stored fold.

Three modes:

    # fold a benchmark instance for 60 seconds
    python predict.py --seq S5 --seconds 60

    # fold an arbitrary H/P string
    python predict.py --sequence HPHPPHHPHPPHPHHPPHPH --seconds 20

    # re-score a stored result without searching
    python predict.py --score results/S9_submission.json

Scoring reads coordinates back with the pure-Python scorer in ``model.py``,
independently of the compiled kernels that produced them, so a fold that only
looks good because of a bug in the JIT path will not pass here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from datasets import SEQUENCES, encode_sequence
from model import contact_count, radius_gyration2, run_remc, run_sa, validate_coords


def fold(seq_str: str, seconds: float = 30.0, seed: int = 0,
         verbose: bool = True) -> dict:
    """Run alternating SA/REMC on ``seq_str`` for roughly ``seconds``.

    This is the single-process path. It exists for quick interactive folds and
    for tests; reproducing the paper's numbers needs ``train.py``, which runs
    many chains in parallel for hours.
    """
    seq_str = seq_str.strip().upper()
    if not seq_str or set(seq_str) - {"H", "P"}:
        raise ValueError("sequence must be a non-empty string of H and P only")
    n = len(seq_str)
    seq = encode_sequence(seq_str)
    rng = np.random.RandomState(seed)

    best_e = 0
    best_coords = None
    empty = np.zeros((n, 2), dtype=np.int32)
    deadline = time.time() + seconds
    rounds = 0

    while time.time() < deadline:
        rounds += 1
        remaining = deadline - time.time()
        steps = int(max(20_000, min(2_000_000, remaining * 200_000)))
        start = best_coords if best_coords is not None else empty
        e, c = run_sa(seq, n, steps, 5.5, 0.002, int(rng.randint(1, 2**31 - 1)),
                      start, best_coords is not None)
        if e < best_e:
            best_e, best_coords = int(e), np.asarray(c, dtype=np.int32).copy()
            if verbose:
                print(f"[fold] round {rounds} SA: contacts={-best_e}")

        if time.time() >= deadline:
            break
        nrep = min(24, max(8, n // 4))
        e, c = run_remc(seq, n, nrep, max(500, n * 30), 60,
                        int(rng.randint(1, 2**31 - 1)), 0.035, 6.0,
                        best_coords if best_coords is not None else empty,
                        best_coords is not None)
        if e < best_e:
            best_e, best_coords = int(e), np.asarray(c, dtype=np.int32).copy()
            if verbose:
                print(f"[fold] round {rounds} REMC: contacts={-best_e}")

    if best_coords is None:
        raise RuntimeError("search produced no fold; increase --seconds")

    coords = [list(map(int, xy)) for xy in best_coords]
    ok, info = validate_coords(seq_str, coords, verbose=False)
    if not ok:
        raise RuntimeError(f"search produced an invalid fold: {info['error']}")
    return {
        "sequence": seq_str,
        "length": n,
        "lattice": "2D",
        "contacts": int(info["contacts"]),
        "energy": -int(info["contacts"]),
        "radius_gyration2": float(info["radius_gyration2"]),
        "rounds": rounds,
        "coords": coords,
    }


def score(path: str | Path) -> dict:
    """Re-score a stored submission or checkpoint JSON."""
    data = json.loads(Path(path).read_text())
    coords = data.get("coords")
    if coords is None:
        raise ValueError(f"{path}: no 'coords' field")

    seq_str = data.get("sequence")
    sid = data.get("sequence_id")
    if seq_str is None:
        if sid not in SEQUENCES:
            raise ValueError(
                f"{path}: no 'sequence' field and 'sequence_id'={sid!r} is not a "
                "known benchmark id, so the fold cannot be scored"
            )
        seq_str = SEQUENCES[sid]["sequence"]

    ok, info = validate_coords(seq_str, coords, verbose=False)
    out = {
        "file": str(path),
        "sequence_id": sid,
        "length": len(seq_str),
        "valid": bool(ok),
        "contacts": int(info["contacts"]) if ok else None,
        "radius_gyration2": round(float(info["radius_gyration2"]), 4) if ok else None,
        "error": None if ok else info["error"],
    }
    if sid in SEQUENCES:
        tgt = int(SEQUENCES[sid]["best"])
        out["target_contacts"] = tgt
        out["gap"] = tgt - out["contacts"] if ok else None
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--seq", choices=sorted(SEQUENCES), help="benchmark instance id")
    g.add_argument("--sequence", help="an arbitrary H/P string to fold")
    g.add_argument("--score", metavar="JSON",
                   help="re-score an existing submission or checkpoint file")
    p.add_argument("--seconds", type=float, default=30.0,
                   help="search budget in seconds (default: 30)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="write the resulting fold to this JSON path")
    args = p.parse_args(argv)

    if args.score:
        result = score(args.score)
        print(json.dumps(result, indent=2))
        return 0 if result["valid"] else 1

    seq_str = SEQUENCES[args.seq]["sequence"] if args.seq else args.sequence
    result = fold(seq_str, seconds=args.seconds, seed=args.seed)
    if args.seq:
        result["sequence_id"] = args.seq
        result["target_contacts"] = int(SEQUENCES[args.seq]["best"])
        result["gap"] = result["target_contacts"] - result["contacts"]

    print(f"\ncontacts = {result['contacts']}"
          + (f"  (best known {result['target_contacts']}, "
             f"gap {result['gap']})" if args.seq else ""))
    print(f"Rg^2     = {result['radius_gyration2']:.4f}")

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
