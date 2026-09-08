<p align="center">
  <img src="assets/banner.png" alt="Revisiting HP Protein Folding — move-set ablation and grid-accelerated pivots for REMC on the 2D square lattice" width="100%">
</p>

<p align="center">
  <a href="https://github.com/OWNER/REPO/actions/workflows/ci.yml"><img src="https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://openreview.net/forum?id=yfMz2KaNJp"><img src="https://img.shields.io/badge/paper-OpenReview-b31b1b.svg" alt="Paper"></a>
  <img src="https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-3776ab.svg" alt="Python 3.10-3.12">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT license"></a>
  <img src="https://img.shields.io/badge/best--known%20contacts-212%2F225%20(94.2%25)-1f9d72.svg" alt="212/225 best-known contacts">
</p>

# Revisiting HP Protein Folding

**Move-Set Ablation and Grid-Accelerated Pivots for REMC on the 2D Square Lattice**

A replica-exchange Monte Carlo solver for the 2D HP protein folding problem, combining
six move operators with grid-based pivot collision detection. On the CAISc 2026
benchmark suite (S5–S10, lengths 48–100) it matches the best-known 23 contacts on S5
and reaches **212 of 225 best-known contacts (94.2%)** in a single 10-hour run on one
CPU node.

The paper's headline contribution is not the solver but the **ablation**: the first
controlled measurement of what each move operator actually contributes. Pivot rotations
alone account for roughly 80% of the total improvement over local-move-only search.

📄 **Paper:** [OpenReview](https://openreview.net/forum?id=yfMz2KaNJp) ·
[PDF](https://openreview.net/pdf?id=yfMz2KaNJp) ·
[local copy](caisc2026_final_paper.pdf)
🏆 **Benchmark:** [CAISc 2026 Verifiable Problems — HP Protein Folding](https://caisc2026.github.io/verifiable-problems/?problem=hp-protein-folding)

---

## The problem

Given a chain of hydrophobic (**H**) and polar (**P**) residues, find a self-avoiding
walk on ℤ² that maximises the number of non-sequential H–H lattice contacts. The
problem is NP-complete on the 2D square lattice, and the search space grows as roughly
μⁿ with μ ≈ 2.638 — about 10⁴² conformations for a 100-residue chain. For chains longer
than 36 residues only heuristic best-known values exist.

Energy is `E = −(contacts)`, so lower is better and the ground state is the
maximum-contact fold.

---

## Results

Every number below is recomputed from the committed coordinate files by
[`analyze.py`](analyze.py), never copied from the solver's own output. CI re-derives
the whole table on every push and fails if a single contact count drifts.

| ID | Length | H% | Best known | Ours | Gap | Rg² | Recorded time |
|---|---|---|---|---|---|---|---|
| S5 | 48 | 52.1% | 23 | **23** | **matched** | 8.1 | 48 min |
| S6 | 50 | 44.0% | 21 | **19** | 2 | 10.4 | 48 min |
| S7 | 60 | 71.7% | 36 | **35** | 1 | 10.5 | 67 min |
| S8 | 64 | 65.6% | 42 | **38** | 4 | 13.1 | 79 min |
| S9 | 85 | 69.4% | 53 | **51** | 2 | 14.8 | 171 min |
| S10 | 100 | 56.0% | 50 | **46** | 4 | 18.7 | 11 min |
| **Total** | | | **225** | **212** | **94.2%** | | |

S1–S4 (lengths 20–36) have optima confirmed by exhaustive search and are used only to
validate the scorer; the solver reaches all four.

> **On the time column.** This is the `elapsed_seconds` field recorded in each
> checkpoint. For S5–S9, which ran to their deadlines, it is the full time allocated to
> that sequence. For S10 it is the time at which its best fold was *found* — the run
> was stopped before its deadline, and the training log shows S10 actually searching
> for about 90 minutes of its 184-minute allocation. See [Provenance](#provenance).

Reproduce the table yourself from the committed coordinates:

```bash
python analyze.py --check
```

---

## Method

The solver alternates parallel simulated annealing and replica-exchange Monte Carlo
rounds inside a wall-clock budget loop, reseeding each round from the best conformation
found so far. Six operators, all preserving the self-avoiding-walk constraint, are
accepted or rejected by the Metropolis criterion `P = min(1, e^(−ΔE/T))`.

| Operator | Budget | What it does | Cost |
|---|---|---|---|
| Corner flip | 12% | Reflects an L-shaped residue across the diagonal | O(n), incremental ΔE |
| End flip | 8% | Swings a terminal residue to a free neighbour of its anchor | O(n), incremental ΔE |
| Simple pull | 8% | Moves an interior residue to a site adjacent to both neighbours | O(n), incremental ΔE |
| Crankshaft | 15% | Swaps a bonded pair to complementary diagonal corners | O(n²) — **see below** |
| **Pivot rotation** | **35%** | Rotates an entire segment 90/180/270° about a pivot residue | O(s) collision + O(n²) ΔE |
| Bond rebridging | 22% | Reverses an internal segment between spliceable bonds | O(n²) — **see below** |

Annealing cools geometrically from `T = 5.5` to `T = 0.002`. REMC uses a geometric
ladder from `T = 0.035` to `T = 6.0` across 12–40 replicas, with even/odd alternating
neighbour swaps at `P = min(1, exp[(βᵢ−βⱼ)(Eⱼ−Eᵢ)])`; the coldest replica is seeded with
the global best. Energy is tracked incrementally but recomputed from scratch every
`N/40` steps, because accumulated ΔE drifts.

### Grid-accelerated pivots

A naive pivot checks each rotated position against every non-rotated residue, at
`O(n · s)` for a segment of length `s`. Since the solver already maintains an occupancy
grid, the check becomes a constant-time lookup per residue:

```
lift the segment out of the grid          O(s)
for each rotated position:
    if grid[position] is occupied -> undo and reject
    else mark it occupied                 O(1) each
ΔE <- E_full(after) - E_full(before)      O(n²)
```

A rotation is a rigid isometry, so it preserves all pairwise distances: a self-avoiding
segment stays self-avoiding, and only collisions against the rest of the chain need
checking. That drops collision detection from `O(n · s)` to `O(s)`.

The measured end-to-end gain is only **1.4–1.5×**, because the `O(n²)` energy recompute
— shared by both versions — dominates total pivot cost. An incremental ΔE for
multi-residue moves would amplify this considerably.

<p align="center">
  <img src="assets/fig2_grid_pivot_speedup.png" alt="Wall-clock time per 200K SA steps, naive versus grid-based pivot collision detection" width="47%">
  <img src="assets/fig3_acceptance_rates.png" alt="Move acceptance rates by operator and sequence" width="47%">
</p>

Pivots have the *lowest* acceptance rate (6–10%) — most rotated segments collide — yet
accepted pivots produce the largest structural changes. End flips have the highest
acceptance (25–64%) and contribute least.

---

## The ablation

Four solver variants, each run for 500K SA steps with 10 independent seeds:

![Move operator ablation: No Pivot, Pivot Only, Pivot + Pull, and Full System across S5, S7 and S9](assets/fig1_move_ablation.png)

| Variant | S5 best (mean) | S7 best (mean) | S9 best (mean) |
|---|---|---|---|
| No Pivot | 15 (11.6) | 24 (20.8) | 29 (25.3) |
| Pivot Only | 19 (17.0) | 32 (29.8) | 43 (37.9) |
| Pivot + Pull | 22 (18.3) | 32 (30.7) | 45 (40.7) |
| Full System | 20 (17.8) | 33 (30.6) | 46 (41.6) |

On S9 (n = 85), removing pivots drops mean contacts from **41.6 to 25.3**, a 39%
decrease. Adding pivots alone recovers 12.6 of the 16.3 total improvement — **77%**.
Local moves contribute the remaining 3.7 by fine-tuning residue positions within the
topology that pivots discovered.

Note that "Pivot + Pull" *beats* "Full System" on S5 (22 vs 20). The paper reads this as
shorter chains favouring simpler move sets. The dead-operator finding below suggests a
more direct explanation.

### Search dynamics

![Best contacts found versus SA progress for S5, S7 and S9](assets/fig4_search_dynamics.png)

Roughly 80% of the final contacts are found in the first 30% of steps. Under a fixed
compute budget, many short runs likely beat few long ones — which is why `train.py`
launches up to 256 parallel chains per round rather than one long chain.

---

## Two dead move operators

**This is a defect in the published solver, reproduced here rather than fixed.**

While building this repository, `crankshaft` and `rebridging` were found to fire
**zero times** in 2000 attempts each — from random walks and from compact annealed
folds alike. Both preconditions are unsatisfiable on the square lattice, for the same
parity reason:

> A chain bond flips the parity of `x + y`. So for residues *i* and *j*, the Manhattan
> distance `|cᵢ − cⱼ|` always has the same parity as `j − i`.

- **Crankshaft** requires `c[i−1]` and `c[i+2]` to be *diagonally* adjacent. Their chain
  separation is 3 (odd), so their Manhattan distance must be odd — but diagonal
  adjacency means a distance of exactly 2. Unsatisfiable.
- **Rebridging** picks `j` as a *lattice neighbour* of `i`, which forces `|i − j|` odd.
  It then requires `c[j−1]` adjacent to `c[i]`, a separation of `j−1−i` (even), so that
  distance must be even and can never be 1. Unsatisfiable.

Confirmed by exhaustive enumeration over all 189,948 self-avoiding walks up to length
12: zero hits for either condition. Both proofs are pinned by tests
([`test_model.py`](test_model.py)) that fail if either operator ever fires.

**Consequences.** The paper states that 72% of the move budget goes to large-scale
structural change. In practice only the 35% pivot share does; crankshaft (15%) plus
rebridging (22%) = **37% of every run was spent on no-ops**. This also gives a simpler
reading of the S5 ablation anomaly above: "Full System" does not lose to "Pivot + Pull"
because a richer move set explores less efficiently, but because it throws away 37% of
its attempts.

This is the *same class of defect* as the cascade-pull bug the paper documents in
[Section 5](caisc2026_final_paper.pdf) — a move whose geometric precondition cannot hold,
failing silently rather than loudly. That one was caught before publication; these two
were not. Since the published results came from exactly this code, the operators are
left intact so the repository reproduces them.

**If you fix them,** expect the two pinning tests to fail, and re-run `train.py` before
updating any number in this README. A corrected crankshaft requires `c[i−1]` and `c[i+2]`
to be lattice-*adjacent*, not diagonal.

---

## Best conformations

Green = hydrophobic (H), blue = polar (P), dashed red = H–H contacts. The compact
diamond-shaped hydrophobic cores in S7 and S9 are the signature of near-optimal folds:
H residues pack tightly in the interior while P segments form the exterior boundary.

<p align="center">
  <img src="assets/fig5_S7_fold.jpg" alt="S7 best conformation, 35 of 36 contacts" width="47%">
  <img src="assets/fig5_S9_fold.jpg" alt="S9 best conformation, 51 of 53 contacts" width="47%">
</p>
<p align="center">
  <img src="assets/fig6_S6_fold.jpg" alt="S6 best conformation, 19 of 21 contacts" width="47%">
  <img src="assets/fig6_S8_fold.jpg" alt="S8 best conformation, 38 of 42 contacts" width="47%">
</p>

- **S5 (48-mer, matched).** Reaches 23 contacts consistently. Balanced 52% H-density
  gives a landscape where pivots efficiently locate the optimal core topology.
- **S7 (60-mer, gap 1).** A dense core spanning the full lattice extent. The last
  contact likely needs a global rearrangement that neither pivots nor the (dead)
  rebridging found.
- **S9 (85-mer, gap 2).** 51 of 53 contacts, the largest compute allocation. The two
  missing contacts suggest the P-rich tail (residues 64–85) is not optimally threaded
  through the core boundary.
- **S8 (64-mer, gap 4).** Highest H-density at 66%, producing a rugged landscape with
  many competing near-optimal core arrangements.
- **S10 (100-mer, gap 4).** The largest chain. Its run was cut short — see
  [Provenance](#provenance).

Redraw any of these from the committed coordinates:

```bash
python analyze.py --figures
```

---

## Install

Requires Python 3.10–3.12. Numba does the heavy lifting; the solver is CPU-only and does
not use a GPU.

```bash
git clone https://github.com/OWNER/REPO.git
cd REPO
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Verify the install against the committed results:

```bash
pytest -q            # 41 tests, ~20 s
python analyze.py    # prints the results table
```

---

## Reproduce

Re-derive every published number from the committed coordinate files — no search, a few
seconds:

```bash
python analyze.py --check              # re-score all six folds, exit non-zero on any error
python analyze.py --markdown           # regenerate the results table above
python analyze.py --figures --banner   # regenerate conformation figures and the banner
```

Check the benchmark sequences against the live competition JSON:

```bash
python datasets.py                                   # print the benchmark
python -c "import datasets; datasets.check_official_problem_data()"
```

---

## Train

`train.py` runs the full search. It splits the budget across sequences by
`TIME_WEIGHTS`, checkpoints every improvement immediately, and refuses to start if the
local sequence snapshot disagrees with the live competition JSON.

```bash
# the published configuration: all six instances, 10 hours
python train.py --hours 10

# a single instance
python train.py --seq S5 --hours 1

# ~1 minute end-to-end check of the whole pipeline
python train.py --smoke --seq S1 --no-strict-official-check
```

| Flag | Default | Meaning |
|---|---|---|
| `--hours` | `9.0` | Total wall-clock budget across all sequences |
| `--seq` | `S5 … S10` | Which instances to solve |
| `--workers` | `min(cpu_count, 16)` | Parallel SA/REMC processes |
| `--no-resume` | off | Ignore existing checkpoints and start cold |
| `--no-strict-official-check` | off | Warn instead of aborting if the live JSON differs |
| `--smoke` | off | Tiny run for CI and sanity checks |

Checkpoints land in `checkpoints/best_<id>.json` and submissions in
`results/<id>_submission.json`. **Running `train.py` from a clean checkout will overwrite
the committed results** — work on a copy if you want to keep them.

> Reproducing the published numbers needs the published budget: 10 hours on a 16-worker
> CPU node. A short run will not match the table.

---

## Inference

```bash
# fold a benchmark instance for 60 seconds
python predict.py --seq S5 --seconds 60

# fold an arbitrary H/P chain
python predict.py --sequence HPHPPHHPHPPHPHHPPHPH --seconds 20 --out fold.json

# re-score a stored fold without searching
python predict.py --score results/S9_submission.json
```

Scoring reads coordinates back with the pure-Python scorer in `model.py`, independently
of the compiled kernels that produced them — a fold that only looks good because of a
bug in the JIT path will not pass.

```json
{
  "file": "results/S9_submission.json",
  "sequence_id": "S9",
  "length": 85,
  "valid": true,
  "contacts": 51,
  "radius_gyration2": 14.8152,
  "target_contacts": 53,
  "gap": 2
}
```

Use the library directly:

```python
from datasets import SEQUENCES
from predict import fold, score

result = fold(SEQUENCES["S5"]["sequence"], seconds=30)
print(result["contacts"], result["coords"][:3])

print(score("results/S7_submission.json"))
```

---

## Repository layout

```
model.py           HP lattice model: energy, six move operators, SA and REMC kernels
datasets.py        Benchmark sequences S1-S10, targets, live-JSON verification
train.py           Search driver: budgeted SA/REMC rounds, checkpointing, submissions
predict.py         Fold a sequence, or re-score a stored fold
analyze.py         Rebuild the results table, figures and banner from coordinates
test_model.py      41 tests: model invariants, dead-operator proofs, result integrity

assets/            Paper figures at source resolution, plus the banner
results/           Submission coordinates (S5-S10) and the verification artifact
checkpoints/       Best fold per sequence, with contacts, Rg² and timing
logs/              Verbatim stdout from the published 10-hour run

caisc2026_final_paper.pdf     The paper
foldpaper_a100_final.ipynb    The original notebook the published run was launched from
```

---

## Provenance

Everything in `results/`, `checkpoints/` and `logs/` comes from the single 10-hour run
on a Lightning AI A100-40GB instance (16 CPU workers; the GPU is unused). Nothing has
been regenerated or retouched, with the exceptions noted here.

- **`logs/train_a100_10h.log`** is the verbatim stdout of the notebook's launch cell,
  reassembled from its two output streams in order. It ends mid-S10 at round 18 because
  the notebook was stopped before the budget expired.
- **`results/S10_submission.json`** is the one file the interrupted run never wrote. It
  was regenerated from `checkpoints/best_S10.json` through the normal
  `train.write_submission` path, which re-validates the fold before writing. Its 46
  contacts are the run's own result, not a re-search.
- **`results/verification_summary_2026-05-29_superseded.json`** is a **pre-fix artifact,
  kept deliberately**. It predates the sequence-verification fix and contains two chains
  that are *not* the benchmark chains — a 63-residue S7 (the real one is 60) and a
  different S10. It is the evidence for the "sequence hallucination" failure mode in
  Section 5 of the paper. **Do not use it for results**; the authoritative artifact is
  `results/verification.json`.
- **H-density** in the results table is computed from the sequences rather than taken
  from the paper's Table 2, which rounds several entries loosely (S7 is 71.7%, printed
  as 67%; S1 is 50.0%, printed as 55%).

The two source materials — the paper PDF and the original notebook — are committed
unmodified. The `.py` modules are that notebook refactored into importable form: the
kernels, move operators, temperature schedules and move probabilities are unchanged, so
the search behaves identically. The notebook's `pip install numba` bootstrap was dropped
in favour of `requirements.txt`.

---

## Limitations

- Benchmark results come from a **single 10-hour run**; no error bars. The ablation
  (Table 1) reports best and mean over 10 seeds.
- 37% of the move budget is dead code, as described above.
- Pivot ΔE is a full `O(n²)` recompute, which caps the grid speedup at ~1.5×. An
  incremental ΔE for multi-residue moves is the single highest-value fix.
- The remaining gap to Wang–Landau on S9 (51 vs 53) is attributable to its
  flat-histogram property, which guarantees exploration of all energy levels — something
  REMC with a geometric ladder cannot match.

---

## Citation

<!-- TODO: replace AUTHOR_NAME with the author list exactly as it appears on the
     OpenReview forum page. The submission PDF is anonymised, so the name is not
     recoverable from it. Also update CITATION.cff and LICENSE. -->

```bibtex
@inproceedings{AUTHOR_NAME2026hpfolding,
  title     = {Revisiting {HP} Protein Folding: Move-Set Ablation and
               Grid-Accelerated Pivots for {REMC} on the {2D} Square Lattice},
  author    = {AUTHOR_NAME},
  booktitle = {Proceedings of the 1st Conference for AI Scientists (CAISc)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=yfMz2KaNJp}
}
```

## References

- Lau & Dill (1989). *A lattice statistical mechanics model of the conformational and sequence spaces of proteins.* Macromolecules 22(10):3986–3997.
- Crescenzi, Goldman, Papadimitriou, Piccolboni & Yannakakis (1998). *On the complexity of protein folding.* J. Comput. Biol. 5(3):423–465.
- Lesh, Mitzenmacher & Whitesides (2003). *A complete and effective move set for simplified protein folding.* RECOMB, 188–195.
- Thachuk, Shmygelska & Hoos (2007). *A replica exchange Monte Carlo algorithm for protein folding in the HP model.* BMC Bioinformatics 8:342.
- Wüst & Landau (2012). *Optimized Wang–Landau sampling of lattice polymers.* J. Chem. Phys. 137:064903.
- Shatabda, Newton, Rashid & Sattar (2013). *An efficient encoding for simplified protein structure prediction using genetic algorithms.* IEEE CEC, 1217–1224.
- Khandoker, Inack & Hibat-Allah (2025). *Lattice protein folding with variational annealing.* Mach. Learn.: Sci. Technol. 6(3):035023.
- Lam, Pitrou & Seibert (2015). *Numba: A LLVM-based Python JIT compiler.* LLVM-HPC @ SC15.

## License

[MIT](LICENSE).
