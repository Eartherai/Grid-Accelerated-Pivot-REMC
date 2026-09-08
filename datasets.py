"""CAISc 2026 HP protein folding benchmark: sequences, targets, and verification.

The ten benchmark instances S1-S10 are held as a hard-coded snapshot of the
official problem JSON, taken on 2026-05-29 from

    https://caisc2026.github.io/verifiable-problems/hp-protein-folding.json

The snapshot exists so that runs are reproducible offline, but it is never
trusted blindly: ``check_official_problem_data`` re-fetches the live JSON and
refuses to continue if a single sequence or target has drifted.

That check is not decoration. An earlier version of this solver ran against two
sequences that had been reconstructed from memory rather than read from the
source, and reported improvements on chains that were not the benchmark chains.
The runtime check is what caught it. See the "Sequence hallucination" note in
Section 5 of the paper.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Dict, Iterable

import numpy as np

OFFICIAL_PROBLEM_URL = (
    "https://caisc2026.github.io/verifiable-problems/hp-protein-folding.json"
)

#: Date the local snapshot below was taken from ``OFFICIAL_PROBLEM_URL``.
SNAPSHOT_DATE = "2026-05-29"

SEQUENCES: Dict[str, dict] = {
    "S1": {
        "sequence": "HPHPPHHPHPPHPHHPPHPH",
        "best": 9,
        "submittable": False,
        "optimal": True,
    },
    "S2": {
        "sequence": "HHPPHPPHPPHPPHPPHPPHPPHH",
        "best": 9,
        "submittable": False,
        "optimal": True,
    },
    "S3": {
        "sequence": "PPHPPHHPPPPHHPPPPHHPPPPHH",
        "best": 8,
        "submittable": False,
        "optimal": True,
    },
    "S4": {
        "sequence": "PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPP",
        "best": 14,
        "submittable": False,
        "optimal": True,
    },
    "S5": {
        "sequence": "PPHPPHHPPHHPPPPPHHHHHHHHHHPPPPPPHHPPHHPPHPPHHHHH",
        "best": 23,
        "submittable": True,
        "optimal": False,
    },
    "S6": {
        "sequence": "PPHPPHPHPHHHHPHPPPHPPPHPPPPHPPPHPPPHPHHHHPHPHPHPHH",
        "best": 21,
        "submittable": True,
        "optimal": False,
    },
    "S7": {
        "sequence": "PPHHHPHHHHHHHHPPPHHHHHHHHHHPHPPPHHHHHHHHHHHHPPPPHHHHHHPHHPHP",
        "best": 36,
        "submittable": True,
        "optimal": False,
    },
    "S8": {
        "sequence": "HHHHHHHHHHHHPHPHPPHHPPHHPPHPPHHPPHHPPHPPHHPPHHPPHPHPHHHHHHHHHHHH",
        "best": 42,
        "submittable": True,
        "optimal": False,
    },
    "S9": {
        "sequence": (
            "HHHHPPPPHHHHHHHHHHHHPPPPPPHHHHHHHHHHHHPPPHHHHHHHHHHHH"
            "PPPHHHHHHHHHHHHPPPHPPHHPPHHPPHPH"
        ),
        "best": 53,
        "submittable": True,
        "optimal": False,
    },
    "S10": {
        "sequence": (
            "PPPHHPPHHHHPPHHHPHHPHHPHHHHPPPPPPPPHHHHHHPPHHHHHHPPPPPPPPP"
            "HPHHPHHHHHHHHHHHPPHHHPHHPHPPHPHHHPPPPPPHHH"
        ),
        "best": 50,
        "submittable": True,
        "optimal": False,
    },
}

#: The six instances with open best-known values; S1-S4 are proven-optimal
#: references used for validation only.
SUBMIT_IDS = ["S5", "S6", "S7", "S8", "S9", "S10"]


def sequence(sid: str) -> str:
    """Return the raw H/P string for ``sid``."""
    return SEQUENCES[sid]["sequence"]


def target(sid: str) -> int:
    """Return the best-known contact count for ``sid``."""
    return int(SEQUENCES[sid]["best"])


def h_fraction(sid: str) -> float:
    """Fraction of residues that are hydrophobic."""
    seq = sequence(sid)
    return seq.count("H") / len(seq)


def encode_sequence(seq_str: str) -> np.ndarray:
    """Encode an H/P string as int8, with H=1 and P=0."""
    return np.array([1 if c == "H" else 0 for c in seq_str], dtype=np.int8)


def check_official_problem_data(strict: bool = True, timeout: int = 10) -> bool:
    """Compare the local snapshot against the live CAISc problem JSON.

    Raises ``RuntimeError`` when ``strict`` and anything differs (or the fetch
    fails); otherwise prints the discrepancy and returns ``False``.
    """
    try:
        with urllib.request.urlopen(OFFICIAL_PROBLEM_URL, timeout=timeout) as f:
            data = json.loads(f.read().decode("utf-8"))
    except Exception as exc:
        msg = f"[official-check] could not fetch official JSON: {exc}"
        if strict:
            raise RuntimeError(msg) from exc
        print(msg)
        return False

    live = {row["id"]: row for row in data["leaderboard"]["rows"]}
    problems = []
    for sid, spec in SEQUENCES.items():
        if sid not in live:
            problems.append(f"{sid}: missing from live data")
            continue
        row = live[sid]
        if row["sequence"] != spec["sequence"]:
            problems.append(f"{sid}: sequence mismatch")
        if int(row["best"]) != int(spec["best"]):
            problems.append(f"{sid}: best mismatch {row['best']} vs {spec['best']}")
        if int(row["length"]) != len(spec["sequence"]):
            problems.append(f"{sid}: length mismatch")

    if problems:
        msg = "official problem data changed:\n" + "\n".join(problems)
        if strict:
            raise RuntimeError(msg)
        print("[official-check]", msg)
        return False

    print("[official-check] OK: local sequences/targets match live CAISc JSON")
    return True


def summary_rows(ids: Iterable[str] = tuple(SEQUENCES)) -> list:
    """Rows of ``(id, length, H%, best_known, proven_optimal)`` for reporting."""
    rows = []
    for sid in ids:
        spec = SEQUENCES[sid]
        seq = spec["sequence"]
        rows.append(
            (sid, len(seq), 100.0 * seq.count("H") / len(seq), int(spec["best"]),
             bool(spec["optimal"]))
        )
    return rows


if __name__ == "__main__":
    print(f"{'ID':<5}{'len':>5}{'H%':>7}{'best':>6}  proven")
    for sid, n, hp, best, opt in summary_rows():
        print(f"{sid:<5}{n:>5}{hp:>6.1f}%{best:>6}  {'yes' if opt else 'no'}")
