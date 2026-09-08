"""Rebuild the results table and conformation figures from stored coordinates.

Everything printed here is recomputed from ``checkpoints/best_*.json`` with the
pure-Python scorer, never copied from a stored ``contacts`` field. If a
coordinate file were edited, the numbers below would change and ``--check``
would fail.

    python analyze.py                 # print the results table
    python analyze.py --check         # exit non-zero if any fold is invalid
    python analyze.py --figures       # redraw assets/fold_<id>.png
    python analyze.py --markdown      # emit the README table
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import SEQUENCES, SUBMIT_IDS
from model import contact_count, radius_gyration2, validate_coords

CHECKPOINT_DIR = Path("checkpoints")
RESULT_DIR = Path("results")
ASSET_DIR = Path("assets")

H_COLOR = "#1f9d72"
P_COLOR = "#3478c7"
CONTACT_COLOR = "#d94841"


def load_folds(ids=tuple(SUBMIT_IDS)) -> list:
    """Load each checkpoint and re-derive every reported quantity from coords."""
    rows = []
    for sid in ids:
        path = CHECKPOINT_DIR / f"best_{sid}.json"
        if not path.exists():
            print(f"[analyze] missing {path}", file=sys.stderr)
            continue
        data = json.loads(path.read_text())
        seq_str = SEQUENCES[sid]["sequence"]

        if data.get("sequence") and data["sequence"] != seq_str:
            raise SystemExit(
                f"{path}: stored sequence does not match the benchmark sequence "
                f"for {sid}. Refusing to report results for a different chain."
            )

        coords = data["coords"]
        ok, info = validate_coords(seq_str, coords, verbose=False)
        target = int(SEQUENCES[sid]["best"])
        contacts = int(info["contacts"]) if ok else None
        rows.append({
            "id": sid,
            "length": len(seq_str),
            "h_pct": 100.0 * seq_str.count("H") / len(seq_str),
            "target": target,
            "contacts": contacts,
            "gap": (target - contacts) if ok else None,
            "rg2": float(info["radius_gyration2"]) if ok else None,
            # For a sequence that ran to its deadline this is the full time
            # allocated to it. For one whose run was cut short it is the time
            # at which its best fold was found. See "Provenance" in the README.
            "elapsed_seconds": float(data.get("elapsed_seconds", 0.0)),
            "valid": bool(ok),
            "error": None if ok else info["error"],
            "coords": coords,
            "sequence": seq_str,
        })
    return rows


def print_table(rows) -> None:
    print(f"{'ID':<5}{'len':>5}{'H%':>7}{'best':>6}{'ours':>6}{'gap':>5}"
          f"{'Rg^2':>8}{'t_rec':>9}  valid")
    print("-" * 60)
    for r in rows:
        t = f"{r['elapsed_seconds']/60:.0f}m" if r["elapsed_seconds"] else "-"
        print(f"{r['id']:<5}{r['length']:>5}{r['h_pct']:>6.1f}%{r['target']:>6}"
              f"{r['contacts'] if r['contacts'] is not None else '-':>6}"
              f"{r['gap'] if r['gap'] is not None else '-':>5}"
              f"{r['rg2']:>8.1f}{t:>9}  {'yes' if r['valid'] else 'NO'}")
    ours = sum(r["contacts"] for r in rows if r["contacts"] is not None)
    tgt = sum(r["target"] for r in rows)
    print("-" * 60)
    print(f"total {ours}/{tgt} = {100.0*ours/tgt:.1f}% of best known")


def markdown_table(rows) -> str:
    out = ["| ID | Length | H% | Best known | Ours | Gap | Rg² | Recorded time |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        status = "**matched**" if r["gap"] == 0 else f"{r['gap']}"
        t = f"{r['elapsed_seconds']/60:.0f} min" if r["elapsed_seconds"] else "—"
        out.append(f"| {r['id']} | {r['length']} | {r['h_pct']:.1f}% | "
                   f"{r['target']} | **{r['contacts']}** | {status} | "
                   f"{r['rg2']:.1f} | {t} |")
    ours = sum(r["contacts"] for r in rows if r["contacts"] is not None)
    tgt = sum(r["target"] for r in rows)
    out.append(f"| **Total** | | | **{tgt}** | **{ours}** | "
               f"**{100.0*ours/tgt:.1f}%** | | |")
    return "\n".join(out)


def plot_fold(row, out_dir: Path = ASSET_DIR, dpi: int = 300) -> Path | None:
    """Redraw one conformation: chain in grey, H-H contacts dashed red."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
    except ImportError:
        print("[analyze] matplotlib not installed; skipping figures",
              file=sys.stderr)
        return None

    import numpy as np
    seq_str = row["sequence"]
    coords = np.asarray(row["coords"], dtype=int)
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.set_aspect("equal")
    ax.plot(coords[:, 0], coords[:, 1], "-", color="#b8b8b8", lw=1.5, zorder=1)
    for i in range(len(seq_str)):
        if seq_str[i] != "H":
            continue
        for j in range(i + 2, len(seq_str)):
            if seq_str[j] != "H":
                continue
            if abs(coords[j, 0] - coords[i, 0]) + \
               abs(coords[j, 1] - coords[i, 1]) == 1:
                ax.plot([coords[i, 0], coords[j, 0]],
                        [coords[i, 1], coords[j, 1]], "--",
                        color=CONTACT_COLOR, lw=1.7, alpha=0.55, zorder=2)
    for i, (x, y) in enumerate(coords):
        ax.add_patch(Circle((x, y), 0.35,
                            fc=H_COLOR if seq_str[i] == "H" else P_COLOR,
                            ec="white", lw=1.2, zorder=3))
        ax.text(x, y, str(i), ha="center", va="center", fontsize=4.5,
                color="white", zorder=4)
    gap = row["gap"]
    status = "MATCHED_BEST_KNOWN" if gap == 0 else f"GAP_{gap}"
    ax.set_title(f"{row['id']}: contacts={row['contacts']} "
                 f"target={row['target']} {status}", fontsize=10)
    pad = 2
    ax.set_xlim(coords[:, 0].min() - pad, coords[:, 0].max() + pad)
    ax.set_ylim(coords[:, 1].min() - pad, coords[:, 1].max() + pad)
    ax.grid(True, alpha=0.13)
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"fold_{row['id']}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[analyze] wrote {path}")
    return path


def plot_banner(rows, out_path: Path = ASSET_DIR / "banner.png", dpi: int = 200):
    """Render the README banner from the real S9 fold.

    Nothing here is decoration: the chain drawn on the right is the committed
    85-mer conformation, and the contact count under it is recomputed from
    those coordinates like every other number in this file.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
    except ImportError:
        print("[analyze] matplotlib not installed; skipping banner",
              file=sys.stderr)
        return None

    import numpy as np
    row = next((r for r in rows if r["id"] == "S9"), None)
    if row is None or not row["valid"]:
        print("[analyze] need a valid S9 fold for the banner", file=sys.stderr)
        return None

    bg, fg, muted, accent = "#0d1117", "#e6edf3", "#8b949e", "#58a6ff"
    fig = plt.figure(figsize=(12, 3.0), facecolor=bg)

    ax = fig.add_axes([0.05, 0.0, 0.58, 1.0])
    ax.set_facecolor(bg)
    ax.axis("off")
    ax.text(0, 0.795, "Revisiting HP Protein Folding", color=fg,
            fontsize=26, fontweight="bold", va="center", ha="left")
    ax.text(0, 0.615, "Move-Set Ablation and Grid-Accelerated Pivots",
            color=fg, fontsize=14, va="center", ha="left")
    ax.text(0, 0.485, "for REMC on the 2D Square Lattice",
            color=fg, fontsize=14, va="center", ha="left")
    ax.plot([0, 0.30], [0.345, 0.345], color=accent, lw=2.2,
            solid_capstyle="butt")
    total = sum(r["contacts"] for r in rows if r["contacts"] is not None)
    tgt = sum(r["target"] for r in rows)
    ax.text(0, 0.205, f"CAISc 2026   ·   {total}/{tgt} best-known contacts "
                      f"({100.0*total/tgt:.1f}%)   ·   S5 matched",
            color=muted, fontsize=11.5, va="center", ha="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax2 = fig.add_axes([0.635, 0.03, 0.35, 0.94])
    ax2.set_facecolor(bg)
    ax2.set_aspect("equal")
    ax2.axis("off")
    seq_str = row["sequence"]
    coords = np.asarray(row["coords"], dtype=int)
    ax2.plot(coords[:, 0], coords[:, 1], "-", color="#30363d", lw=2.0, zorder=1)
    for i in range(len(seq_str)):
        if seq_str[i] != "H":
            continue
        for j in range(i + 2, len(seq_str)):
            if seq_str[j] != "H":
                continue
            if abs(coords[j, 0] - coords[i, 0]) + \
               abs(coords[j, 1] - coords[i, 1]) == 1:
                ax2.plot([coords[i, 0], coords[j, 0]],
                         [coords[i, 1], coords[j, 1]], "--",
                         color=CONTACT_COLOR, lw=1.6, alpha=0.8, zorder=2)
    for i, (x, y) in enumerate(coords):
        ax2.add_patch(Circle((x, y), 0.34,
                             fc=H_COLOR if seq_str[i] == "H" else P_COLOR,
                             ec=bg, lw=1.0, zorder=3))
    pad = 0.8
    ax2.set_xlim(coords[:, 0].min() - pad, coords[:, 0].max() + pad)
    ax2.set_ylim(coords[:, 1].min() - 2.6, coords[:, 1].max() + pad)
    ax2.text(0.5, 0.0, f"S9  ·  85-mer  ·  {row['contacts']}/{row['target']} contacts",
             transform=ax2.transAxes, color=muted, fontsize=10,
             ha="center", va="bottom")

    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=bg)
    plt.close(fig)
    print(f"[analyze] wrote {out_path}")
    return out_path


def write_verification(rows, out_path: Path = RESULT_DIR / "verification.json"):
    """Write the consolidated verification artifact from re-scored folds."""
    out_path.parent.mkdir(exist_ok=True)
    payload = {
        "conference": "CAISc 2026",
        "track": "Verifiable Problems",
        "problem": "HP Protein Folding",
        "note": ("Every field below is recomputed from the coordinate files by "
                 "analyze.py, not copied from the solver's own output."),
        "results": [
            {k: v for k, v in r.items() if k != "coords"} | {"coords": r["coords"]}
            for r in rows
        ],
        "total_contacts": sum(r["contacts"] for r in rows
                              if r["contacts"] is not None),
        "total_best_known": sum(r["target"] for r in rows),
    }
    payload["percent_of_best_known"] = round(
        100.0 * payload["total_contacts"] / payload["total_best_known"], 2)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[analyze] wrote {out_path}")
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--check", action="store_true",
                   help="exit non-zero if any stored fold is invalid")
    p.add_argument("--figures", action="store_true",
                   help="redraw conformation figures into assets/")
    p.add_argument("--markdown", action="store_true",
                   help="print the results table as markdown")
    p.add_argument("--banner", action="store_true",
                   help="render assets/banner.png from the S9 fold")
    p.add_argument("--write-verification", action="store_true",
                   help="write results/verification.json")
    p.add_argument("--seq", nargs="*", default=list(SUBMIT_IDS))
    args = p.parse_args(argv)

    rows = load_folds(tuple(args.seq))
    if not rows:
        print("[analyze] no checkpoints found", file=sys.stderr)
        return 1

    if args.markdown:
        print(markdown_table(rows))
    else:
        print_table(rows)

    if args.figures:
        for r in rows:
            if r["valid"]:
                plot_fold(r)

    if args.banner:
        plot_banner(rows)

    if args.write_verification:
        write_verification(rows)

    if args.check:
        bad = [r for r in rows if not r["valid"]]
        if bad:
            for r in bad:
                print(f"[analyze] INVALID {r['id']}: {r['error']}", file=sys.stderr)
            return 1
        print("[analyze] all folds valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
