"""Tests for the lattice model, the move operators, and the stored results.

Run with:  pytest -q

The suite is deliberately split in two. The first half checks the model's
invariants on tiny inputs. The second half re-scores the committed coordinate
files, so the published numbers in the README are covered by CI: if a fold file
is corrupted or a sequence is swapped, these fail.

No network access is required. ``test_snapshot_matches_live_json`` is skipped
unless ``HP_CHECK_LIVE=1`` is set.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

import datasets
from datasets import SEQUENCES, SUBMIT_IDS, encode_sequence
from model import (
    contact_count,
    do_crankshaft,
    do_pivot,
    do_rebridging,
    energy_full,
    init_saw,
    init_snake,
    propose_corner,
    propose_end,
    propose_pull,
    radius_gyration2,
    rebuild_grid_inplace,
    run_remc,
    run_sa,
    valid_walk,
    validate_coords,
)

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
RESULT_DIR = Path(__file__).parent / "results"


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def test_contact_count_on_known_fold():
    # A 2x3 block of H residues folded as a snake. Residues 0 and 3 are diagonal
    # neighbours of each other's partners; the two H-H contacts are (0,3), (2,5).
    seq = "HHHHHH"
    coords = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 2], [1, 2]]
    assert contact_count(seq, coords) == 2
    ok, info = validate_coords(seq, coords, verbose=False)
    assert ok and info["contacts"] == 2


def test_sequential_neighbours_are_not_contacts():
    # Every bonded pair is lattice-adjacent by definition; none may be counted.
    assert contact_count("HHHH", [[0, 0], [1, 0], [2, 0], [3, 0]]) == 0


def test_polar_residues_never_contribute():
    seq = "PPPP"
    coords = [[0, 0], [1, 0], [1, 1], [0, 1]]
    assert contact_count(seq, coords) == 0


def test_energy_full_matches_python_scorer():
    for sid in ("S1", "S4", "S5"):
        seq_str = SEQUENCES[sid]["sequence"]
        seq = encode_sequence(seq_str)
        n = len(seq_str)
        c, _, _, _, ok = init_snake(n)
        assert ok
        assert energy_full(c, seq, n) == -contact_count(seq_str, c.tolist())


def test_radius_gyration_of_symmetric_square():
    # Four points at distance sqrt(2)/2 from the centroid: Rg^2 = 1/2.
    assert radius_gyration2([[0, 0], [1, 0], [1, 1], [0, 1]]) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def test_validate_rejects_self_overlap():
    ok, info = validate_coords("HHHH", [[0, 0], [1, 0], [0, 0], [0, 1]],
                               verbose=False)
    assert not ok and "self-overlap" in info["error"]


def test_validate_rejects_broken_bond():
    ok, info = validate_coords("HHH", [[0, 0], [5, 0], [5, 1]], verbose=False)
    assert not ok and "broken chain bond" in info["error"]


def test_validate_rejects_wrong_length():
    ok, info = validate_coords("HHHH", [[0, 0], [1, 0]], verbose=False)
    assert not ok and "length mismatch" in info["error"]


# --------------------------------------------------------------------------- #
# Initialisation and move operators
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n", [4, 20, 48, 85])
def test_initialisers_produce_valid_walks(n):
    for init in (init_snake, init_saw):
        coords, grid, off, gs, ok = init(n)
        assert ok
        assert valid_walk(coords, n)
        assert rebuild_grid_inplace(coords, grid, n, off, gs)


def test_pivot_preserves_walk_grid_and_energy():
    """Every accepted pivot must leave a SAW, a synced grid, and an honest dE."""
    np.random.seed(7)
    seq_str = SEQUENCES["S5"]["sequence"]
    seq = encode_sequence(seq_str)
    n = len(seq_str)
    coords, grid, off, gs, ok = init_saw(n)
    assert ok

    applied = 0
    for _ in range(400):
        before = energy_full(coords, seq, n)
        accepted, de = do_pivot(coords, grid, seq, n, off, gs)
        if not accepted:
            continue
        applied += 1
        assert valid_walk(coords, n), "pivot broke the walk"
        # The grid must still describe exactly the occupied sites.
        check = np.zeros_like(grid)
        assert rebuild_grid_inplace(coords, check, n, off, gs)
        assert np.array_equal(grid, check), "pivot desynced the grid"
        # The reported delta must match a full recompute.
        assert energy_full(coords, seq, n) - before == de
    assert applied > 0, "no pivot fired; the test proves nothing"


# --- Dead move operators -------------------------------------------------- #
#
# These two tests pin a defect rather than a feature. In the published solver
# both `do_crankshaft` and `do_rebridging` have preconditions that no
# conformation on the square lattice can satisfy, so together they burn 37% of
# the move budget on no-ops. See "Two dead move operators" in the README.
#
# They are kept passing, not fixed, so the repository reproduces the published
# run exactly. Fixing either operator changes the search and invalidates the
# committed results; if you do fix one, expect these two tests to fail, and
# re-run `train.py` before updating any published number.

def test_crankshaft_precondition_is_unsatisfiable():
    """c[i-1] and c[i+2] are 3 bonds apart, so their Manhattan distance is odd;
    diagonal adjacency needs distance 2. The condition can never hold."""
    np.random.seed(13)
    seq_str = SEQUENCES["S5"]["sequence"]
    seq = encode_sequence(seq_str)
    n = len(seq_str)
    coords, grid, off, gs, ok = init_saw(n)
    assert ok
    fired = sum(do_crankshaft(coords, grid, seq, n, off, gs)[0]
                for _ in range(2000))
    assert fired == 0, (
        "do_crankshaft fired, so its precondition is reachable after all. "
        "The dead-operator finding in the README needs revisiting."
    )


def test_rebridging_precondition_is_unsatisfiable():
    """j is picked as a lattice neighbour of i, forcing |i-j| odd; the splice
    check then needs c[j-1] adjacent to c[i] at even separation. Impossible."""
    np.random.seed(17)
    seq_str = SEQUENCES["S9"]["sequence"]
    seq = encode_sequence(seq_str)
    n = len(seq_str)
    coords, grid, off, gs, ok = init_saw(n)
    assert ok
    fired = sum(do_rebridging(coords, grid, seq, n, off, gs)[0]
                for _ in range(2000))
    assert fired == 0, (
        "do_rebridging fired, so its precondition is reachable after all. "
        "The dead-operator finding in the README needs revisiting."
    )


def test_parity_argument_holds_over_all_short_walks():
    """Exhaustive confirmation of the parity argument on every SAW up to n=10.

    A chain bond flips the parity of (x+y), so residues i and j have Manhattan
    distance congruent to (j-i) mod 2. Both dead preconditions ask for a
    distance of the wrong parity.
    """
    steps = ((1, 0), (-1, 0), (0, 1), (0, -1))

    def walks(n):
        def rec(path, seen):
            if len(path) == n:
                yield list(path)
                return
            x, y = path[-1]
            for dx, dy in steps:
                p = (x + dx, y + dy)
                if p in seen:
                    continue
                seen.add(p)
                path.append(p)
                yield from rec(path, seen)
                path.pop()
                seen.remove(p)
        yield from rec([(0, 0)], {(0, 0)})

    crank = rebridge = 0
    for n in range(4, 11):
        for w in walks(n):
            for i in range(1, n - 2):
                (px, py), (qx, qy) = w[i - 1], w[i + 2]
                crank += (abs(px - qx) == 1 and abs(py - qy) == 1)
            for i0 in range(n):
                for dx, dy in steps:
                    t = (w[i0][0] + dx, w[i0][1] + dy)
                    if t not in w:
                        continue
                    j0 = w.index(t)
                    if abs(j0 - i0) <= 2:
                        continue
                    i, j = min(i0, j0), max(i0, j0)
                    if i + 1 >= n or j - 1 < 0:
                        continue
                    rebridge += (abs(w[j - 1][0] - w[i][0])
                                 + abs(w[j - 1][1] - w[i][1]) == 1)
    assert crank == 0, "a walk satisfies the crankshaft precondition"
    assert rebridge == 0, "a walk satisfies the rebridging splice condition"


@pytest.mark.parametrize("propose", [propose_corner, propose_end, propose_pull])
def test_single_residue_proposals_target_free_sites(propose):
    """Proposals must only ever name an unoccupied, in-bounds lattice site."""
    np.random.seed(11)
    seq_str = SEQUENCES["S5"]["sequence"]
    n = len(seq_str)
    coords, grid, off, gs, ok = init_saw(n)
    assert ok

    proposed = 0
    for _ in range(600):
        idx, nx, ny = propose(coords, grid, n, off, gs)
        if idx < 0:
            continue
        proposed += 1
        assert 0 <= idx < n
        assert grid[nx + off, ny + off] == 0
    assert proposed > 0, f"{propose.__name__} never proposed anything"


def test_pivot_rejection_leaves_state_untouched():
    """A rejected pivot must restore the grid exactly, not leave holes behind."""
    np.random.seed(3)
    seq_str = SEQUENCES["S5"]["sequence"]
    seq = encode_sequence(seq_str)
    n = len(seq_str)
    coords, grid, off, gs, ok = init_saw(n)
    assert ok

    rejections = 0
    for _ in range(400):
        coords_before = coords.copy()
        grid_before = grid.copy()
        accepted, _ = do_pivot(coords, grid, seq, n, off, gs)
        if accepted:
            continue
        rejections += 1
        assert np.array_equal(coords, coords_before)
        assert np.array_equal(grid, grid_before)
    assert rejections > 0, "no pivot was rejected; the test proves nothing"


# --------------------------------------------------------------------------- #
# Search drivers
# --------------------------------------------------------------------------- #

def test_sa_returns_consistent_energy_and_valid_fold():
    seq_str = SEQUENCES["S1"]["sequence"]
    seq = encode_sequence(seq_str)
    n = len(seq_str)
    empty = np.zeros((n, 2), dtype=np.int32)
    e, c = run_sa(seq, n, 3000, 2.0, 0.1, 12345, empty, False)
    assert valid_walk(c, n)
    assert e == energy_full(c, seq, n)
    assert e == -contact_count(seq_str, c.tolist())


def test_remc_returns_consistent_energy_and_valid_fold():
    seq_str = SEQUENCES["S1"]["sequence"]
    seq = encode_sequence(seq_str)
    n = len(seq_str)
    empty = np.zeros((n, 2), dtype=np.int32)
    e0, c0 = run_sa(seq, n, 500, 2.0, 0.1, 999, empty, False)
    e, c = run_remc(seq, n, 4, 40, 3, 23456, 0.1, 3.0, c0, True)
    assert valid_walk(c, n)
    assert e == energy_full(c, seq, n)
    assert e <= e0, "REMC seeded from the best fold must not lose ground"


@pytest.mark.parametrize("sid", ["S1", "S2", "S3"])
def test_solver_reaches_proven_optimum(sid):
    """S1-S3 have optima confirmed by exhaustive search; best-of-16 restarts
    reaches them. Single restarts do not reliably, which is the whole reason
    ``train.py`` runs many chains in parallel rather than one long one."""
    seq_str = SEQUENCES[sid]["sequence"]
    seq = encode_sequence(seq_str)
    n = len(seq_str)
    empty = np.zeros((n, 2), dtype=np.int32)
    best = 0
    for seed in range(1, 17):
        e, c = run_sa(seq, n, 120_000, 5.5, 0.002, seed, empty, False)
        e2, _ = run_remc(seq, n, 16, 800, 60, seed * 77 + 1, 0.035, 6.0, c, True)
        best = min(best, int(e), int(e2))
    assert -best == SEQUENCES[sid]["best"]


# --------------------------------------------------------------------------- #
# Benchmark data
# --------------------------------------------------------------------------- #

def test_all_sequences_are_hp_strings():
    for sid, spec in SEQUENCES.items():
        assert set(spec["sequence"]) <= {"H", "P"}, sid
        assert spec["best"] > 0, sid


def test_submit_ids_are_the_open_instances():
    assert SUBMIT_IDS == [s for s, v in SEQUENCES.items() if v["submittable"]]
    assert all(not SEQUENCES[s]["optimal"] for s in SUBMIT_IDS)


@pytest.mark.skipif(os.environ.get("HP_CHECK_LIVE") != "1",
                    reason="set HP_CHECK_LIVE=1 to check the live CAISc JSON")
def test_snapshot_matches_live_json():
    assert datasets.check_official_problem_data(strict=True)


# --------------------------------------------------------------------------- #
# Stored results: these guard the numbers published in the README
# --------------------------------------------------------------------------- #

PUBLISHED = {"S5": 23, "S6": 19, "S7": 35, "S8": 38, "S9": 51, "S10": 46}


@pytest.mark.parametrize("sid", sorted(PUBLISHED))
def test_checkpoint_reproduces_published_contacts(sid):
    path = CHECKPOINT_DIR / f"best_{sid}.json"
    assert path.exists(), f"missing {path}"
    data = json.loads(path.read_text())
    seq_str = SEQUENCES[sid]["sequence"]

    assert data["sequence"] == seq_str, f"{sid}: stored sequence is not the benchmark"
    ok, info = validate_coords(seq_str, data["coords"], verbose=False)
    assert ok, f"{sid}: {info.get('error')}"
    assert info["contacts"] == PUBLISHED[sid]
    assert info["contacts"] == data["contacts"]


@pytest.mark.parametrize("sid", sorted(PUBLISHED))
def test_submission_matches_checkpoint(sid):
    sub = RESULT_DIR / f"{sid}_submission.json"
    assert sub.exists(), f"missing {sub}"
    sub_data = json.loads(sub.read_text())
    ck_data = json.loads((CHECKPOINT_DIR / f"best_{sid}.json").read_text())
    assert sub_data["sequence_id"] == sid
    assert sub_data["lattice"] == "2D"
    assert sub_data["coords"] == ck_data["coords"]


def test_total_matches_the_headline_claim():
    """212 of 225 best-known contacts, i.e. 94.2%, as reported in the paper."""
    total = 0
    for sid in SUBMIT_IDS:
        data = json.loads((CHECKPOINT_DIR / f"best_{sid}.json").read_text())
        total += contact_count(SEQUENCES[sid]["sequence"], data["coords"])
    target = sum(SEQUENCES[s]["best"] for s in SUBMIT_IDS)
    assert total == 212
    assert target == 225
    assert round(100.0 * total / target, 1) == 94.2


def test_s5_matches_best_known():
    data = json.loads((CHECKPOINT_DIR / "best_S5.json").read_text())
    contacts = contact_count(SEQUENCES["S5"]["sequence"], data["coords"])
    assert contacts == SEQUENCES["S5"]["best"] == 23
