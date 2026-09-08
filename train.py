"""Search driver: alternating SA and REMC rounds under a wall-clock budget.

Each sequence gets a slice of the total budget proportional to its weight in
``TIME_WEIGHTS``, then loops:

    seed from best-so-far -> parallel SA chains -> parallel REMC runs -> repeat

until its deadline. Every improvement is written to ``checkpoints/best_<id>.json``
immediately, so an interrupted run keeps everything it found, and ``--resume``
picks up from those files.

Usage
-----
    python train.py --hours 10                  # full run, all six instances
    python train.py --seq S5 S7 --hours 1       # subset
    python train.py --smoke                     # ~1 minute end-to-end check
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing import cpu_count, get_context
from pathlib import Path

import numpy as np

from datasets import (
    OFFICIAL_PROBLEM_URL,
    SEQUENCES,
    SUBMIT_IDS,
    check_official_problem_data,
    encode_sequence,
)
from model import (
    energy_full,
    init_snake,
    run_remc,
    run_sa,
    validate_coords,
    valid_walk,
)

RESULT_DIR = Path("results")
CHECKPOINT_DIR = Path("checkpoints")

#: Budget shares. S9 and S10 get the most because search difficulty grows far
#: faster than chain length: the number of self-avoiding walks is ~mu^n with
#: mu ~ 2.638, so a 100-mer has ~10^42 conformations.
TIME_WEIGHTS = {"S5": 0.8, "S6": 0.8, "S7": 1.1, "S8": 1.3, "S9": 2.8, "S10": 3.0}

#: Annealing schedule, shared by SA and the REMC inner loop.
T_START = 5.5
T_END = 0.002

#: REMC ladder bounds.
REMC_T_MIN = 0.035
REMC_T_MAX = 6.0


@dataclass
class SolverConfig:
    total_hours: float = 9.0
    sequence_ids: tuple = tuple(SUBMIT_IDS)
    max_workers: int = max(1, min(cpu_count(), 16))
    strict_official_check: bool = True
    resume: bool = True
    smoke: bool = False
    random_seed: int = 20260529


# --------------------------------------------------------------------------- #
# Parallel execution
# --------------------------------------------------------------------------- #

def _sa_worker(args):
    return run_sa(*args)


def _remc_worker(args):
    return run_remc(*args)


def _safe_pool_map(func, args, workers):
    """Map over a process pool, degrading to serial rather than failing."""
    if workers <= 1 or len(args) <= 1:
        return [func(a) for a in args]
    try:
        method = "fork" if platform.system() != "Windows" else "spawn"
        ctx = get_context(method)
        with ctx.Pool(processes=workers) as pool:
            return pool.map(func, args)
    except Exception as exc:
        print(f"[pool] multiprocessing failed, falling back to serial: {exc}")
        traceback.print_exc(limit=1)
        return [func(a) for a in args]


# --------------------------------------------------------------------------- #
# Result IO
# --------------------------------------------------------------------------- #

def checkpoint_path(sid: str) -> Path:
    return CHECKPOINT_DIR / f"best_{sid}.json"


def make_result(sid: str, coords, elapsed_seconds: float = 0.0) -> dict:
    """Build a checkpoint record, re-scoring the fold with the Python scorer."""
    spec = SEQUENCES[sid]
    seq_str = spec["sequence"]
    ok, info = validate_coords(seq_str, coords, verbose=False)
    if not ok:
        return {"sequence_id": sid, "valid": False, "error": info["error"],
                "coords": None}
    contacts = int(info["contacts"])
    target = int(spec["best"])
    if contacts > target:
        status = "NEW_RECORD"
    elif contacts == target:
        status = "MATCHED_BEST_KNOWN"
    else:
        status = f"GAP_{target - contacts}"
    return {
        "sequence_id": sid,
        "lattice": "2D",
        "sequence": seq_str,
        "length": len(seq_str),
        "target_contacts": target,
        "contacts": contacts,
        "energy": -contacts,
        "radius_gyration2": float(info["radius_gyration2"]),
        "status": status,
        "elapsed_seconds": float(elapsed_seconds),
        "coords": [list(map(int, xy)) for xy in coords],
    }


def save_checkpoint(result: dict) -> None:
    if not result or not result.get("valid", True):
        return
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    path = checkpoint_path(result["sequence_id"])
    with path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"[checkpoint] {path} contacts={result['contacts']} "
          f"status={result['status']}")


def load_checkpoint(sid: str):
    """Load and re-validate a checkpoint. Invalid files are ignored, not trusted."""
    path = checkpoint_path(sid)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        seq_str = SEQUENCES[sid]["sequence"]
        ok, info = validate_coords(seq_str, data.get("coords"), verbose=False)
        if not ok:
            print(f"[checkpoint] ignoring invalid {path}: {info['error']}")
            return None
        data["contacts"] = int(info["contacts"])
        data["energy"] = -int(info["contacts"])
        data["radius_gyration2"] = float(info["radius_gyration2"])
        print(f"[checkpoint] loaded {sid}: contacts={data['contacts']}")
        return data
    except Exception as exc:
        print(f"[checkpoint] could not load {path}: {exc}")
        return None


def write_submission(result: dict) -> None:
    """Write the coordinates-only submission file, re-validating first."""
    sid = result["sequence_id"]
    if result.get("coords") is None:
        raise ValueError(f"{sid}: no coordinates to submit")
    seq_str = SEQUENCES[sid]["sequence"]
    ok, info = validate_coords(seq_str, result["coords"], verbose=False)
    if not ok:
        raise ValueError(f"{sid}: invalid fold, not writing submission: {info}")
    result["contacts"] = int(info["contacts"])
    result["energy"] = -int(info["contacts"])
    result["radius_gyration2"] = float(info["radius_gyration2"])
    RESULT_DIR.mkdir(exist_ok=True)
    path = RESULT_DIR / f"{sid}_submission.json"
    with path.open("w") as f:
        json.dump({"sequence_id": sid, "lattice": "2D",
                   "coords": result["coords"]}, f, indent=2)
    print(f"[submission] wrote {path}")


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #

def _sequence_hours(config: SolverConfig) -> dict:
    ids = list(config.sequence_ids)
    total_weight = sum(TIME_WEIGHTS.get(sid, 1.0) for sid in ids)
    return {sid: config.total_hours * TIME_WEIGHTS.get(sid, 1.0) / total_weight
            for sid in ids}


def _round_schedule(n: int, workers: int, remaining: float, smoke: bool) -> dict:
    """Pick SA/REMC sizes for one round from the time left and chain length."""
    if smoke:
        return {"sa_chains": max(2, min(workers, 4)), "sa_steps": 1500,
                "remc_runs": max(1, min(workers, 2)), "nrep": 4,
                "spe": 80, "nex": 4}
    sa_chains = min(max(8, workers * 8), 256)
    if n >= 85:
        sa_chains = min(max(6, workers * 6), 192)
    return {
        "sa_chains": sa_chains,
        "sa_steps": min(4_000_000, max(200_000, int(remaining * 1200))),
        "remc_runs": min(max(2, workers), 12),
        "nrep": min(40, max(12, n // 3)),
        "spe": min(8000, max(1000, n * 55)),
        "nex": min(1200, max(100, int(remaining * 0.25))),
    }


def solve_sequence(sid: str, hours: float, workers: int,
                   resume: bool = True, smoke: bool = False) -> dict:
    """Search ``sid`` until its deadline, checkpointing every improvement."""
    if sid not in SEQUENCES:
        raise KeyError(f"unknown sequence id: {sid}")
    spec = SEQUENCES[sid]
    seq_str = spec["sequence"]
    target = int(spec["best"])
    n = len(seq_str)
    seq = encode_sequence(seq_str)

    print("\n" + "=" * 78)
    print(f"[solve] {sid} length={n} target={target} "
          f"budget={hours:.3f}h workers={workers}")
    print("=" * 78)

    best_result = load_checkpoint(sid) if resume else None
    if best_result is not None:
        best_coords = np.asarray(best_result["coords"], dtype=np.int32)
        best_e = -int(best_result["contacts"])
    else:
        best_coords = np.zeros((n, 2), dtype=np.int32)
        best_e = 0

    deadline = time.time() + max(1.0, hours * 3600.0)
    start_time = time.time()
    round_no = 0
    workers = max(1, int(workers))

    while time.time() < deadline:
        remaining = deadline - time.time()
        if smoke and round_no >= 1:
            break
        if remaining < (5.0 if smoke else 35.0):
            break
        round_no += 1
        sched = _round_schedule(n, workers, remaining, smoke)

        print(f"[round {round_no}] SA {sched['sa_chains']}x{sched['sa_steps']} "
              f"then REMC {sched['remc_runs']}x{sched['nrep']}rep x "
              f"{sched['nex']}ex x {sched['spe']}steps "
              f"remaining={remaining/60:.1f}min")

        # --- annealing phase: many independent chains, one seeded from best ---
        seeds = np.random.randint(1, 2**31 - 1, size=sched["sa_chains"])
        empty_start = np.zeros((n, 2), dtype=np.int32)
        sa_args = []
        for i in range(sched["sa_chains"]):
            use_seed = best_e < 0 and i == 0
            sa_args.append((seq, n, sched["sa_steps"], T_START, T_END,
                            int(seeds[i]),
                            best_coords if use_seed else empty_start,
                            bool(use_seed)))
        sa_results = _safe_pool_map(_sa_worker, sa_args,
                                    min(workers, sched["sa_chains"]))
        improved = False
        for e, c in sa_results:
            if e < best_e and valid_walk(c, n):
                best_e = int(e)
                best_coords = np.asarray(c, dtype=np.int32).copy()
                improved = True
        if improved:
            best_result = make_result(sid, best_coords, time.time() - start_time)
            print(f"[round {round_no}] SA improved: contacts={-best_e}")
            save_checkpoint(best_result)
        else:
            print(f"[round {round_no}] SA best so far: contacts={-best_e}")

        if time.time() >= deadline:
            break

        # --- replica exchange phase, coldest replica seeded from best ---
        seeds = np.random.randint(1, 2**31 - 1, size=sched["remc_runs"])
        remc_args = []
        for i in range(sched["remc_runs"]):
            use_seed = best_e < 0 and i == 0
            remc_args.append((seq, n, sched["nrep"], sched["spe"], sched["nex"],
                              int(seeds[i]), REMC_T_MIN, REMC_T_MAX,
                              best_coords if use_seed else empty_start,
                              bool(use_seed)))
        remc_results = _safe_pool_map(_remc_worker, remc_args,
                                      min(workers, sched["remc_runs"]))
        improved = False
        for e, c in remc_results:
            if e < best_e and valid_walk(c, n):
                best_e = int(e)
                best_coords = np.asarray(c, dtype=np.int32).copy()
                improved = True
        if improved:
            best_result = make_result(sid, best_coords, time.time() - start_time)
            print(f"[round {round_no}] REMC improved: contacts={-best_e}")
            save_checkpoint(best_result)
        else:
            print(f"[round {round_no}] REMC best so far: contacts={-best_e}")

        if -best_e > target:
            print(f"[round {round_no}] new record candidate; continuing to improve")
        elif -best_e == target:
            print(f"[round {round_no}] matched best-known; continuing to seek record")

    if best_e >= 0:
        # Nothing found: emit a guaranteed-valid fold so every output path stays
        # exercised rather than silently producing nothing.
        c, _, _, _, _ = init_snake(n)
        best_coords = np.asarray(c, dtype=np.int32)
        best_e = int(energy_full(best_coords, seq, n))

    best_result = make_result(sid, best_coords, time.time() - start_time)
    ok, info = validate_coords(seq_str, best_result["coords"], verbose=True)
    if not ok:
        raise RuntimeError(f"{sid}: invalid final fold: {info}")
    save_checkpoint(best_result)
    write_submission(best_result)
    print(f"[done] {sid}: contacts={best_result['contacts']} "
          f"target={target} status={best_result['status']}")
    return best_result


def write_verification_artifact(results, config: SolverConfig) -> Path:
    RESULT_DIR.mkdir(exist_ok=True)
    artifact = {
        "conference": "CAISc 2026",
        "track": "Verifiable Problems",
        "problem": "HP Protein Folding",
        "problem_url": OFFICIAL_PROBLEM_URL,
        "method": ("SA + replica exchange with corner/end/pull/crankshaft/"
                   "pivot/rebridging moves, grid-accelerated pivots"),
        "gpu_used": False,
        "cpu_workers": int(config.max_workers),
        "total_hours_requested": float(config.total_hours),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "platform": platform.platform(),
        "results": results,
    }
    path = RESULT_DIR / "verification.json"
    with path.open("w") as f:
        json.dump(artifact, f, indent=2)
    print(f"[artifact] wrote {path}")
    return path


def print_summary(results) -> None:
    print("\n" + "=" * 78)
    print("FINAL SUMMARY")
    print("=" * 78)
    print(f"{'ID':<5} {'Len':>4} {'Target':>7} {'Ours':>6} {'Rg2':>10}  Status")
    print("-" * 78)
    for r in results:
        print(f"{r['sequence_id']:<5} {r['length']:>4} {r['target_contacts']:>7} "
              f"{r['contacts']:>6} {r['radius_gyration2']:>10.4f}  {r['status']}")
    print("=" * 78)


def run_competition(config: SolverConfig):
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    check_official_problem_data(strict=config.strict_official_check)

    hours_by_sid = _sequence_hours(config)
    results = []
    for sid in config.sequence_ids:
        results.append(solve_sequence(sid, hours=hours_by_sid[sid],
                                      workers=config.max_workers,
                                      resume=config.resume, smoke=config.smoke))
    write_verification_artifact(results, config)
    print_summary(results)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--hours", type=float, default=9.0,
                        help="total wall-clock budget across all sequences")
    parser.add_argument("--seq", nargs="*", default=SUBMIT_IDS,
                        help="sequence ids to solve (default: S5-S10)")
    parser.add_argument("--workers", type=int,
                        default=max(1, min(cpu_count(), 16)))
    parser.add_argument("--no-strict-official-check", action="store_true",
                        help="warn instead of aborting if the live JSON differs")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore existing checkpoints and start cold")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny end-to-end run for CI and sanity checks")
    args = parser.parse_args(argv)

    return run_competition(SolverConfig(
        total_hours=args.hours,
        sequence_ids=tuple(args.seq),
        max_workers=args.workers,
        strict_official_check=not args.no_strict_official_check,
        resume=not args.no_resume,
        smoke=args.smoke,
    ))


if __name__ == "__main__":
    main()
