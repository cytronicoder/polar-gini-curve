"""Simulate PGC RMSD for random foreground selections (Simulation 1).
Saves per-trial RMSDs, summaries, and example curves + plots.

Run:
  python3 examples/rmsd_simulation.py \
    --n 5000 --angles 360 --repeats 1000 \
    --seed 42 --outdir results
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from polargini.pgc import polar_gini_curve
    from polargini.metrics import rmsd
except ImportError as e:
    print(f"Error importing polargini: {e}")
    sys.exit(1)

PERCENTAGES = np.arange(5, 100, 5)


def sample_unit_disk_rejection(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform unit disk via sampling-by-rejection."""
    pts = []
    while len(pts) < n:
        xy = rng.uniform(-1.0, 1.0, size=(n, 2))
        mask = (xy[:, 0] ** 2 + xy[:, 1] ** 2) <= 1.0
        pts.extend(xy[mask].tolist())
    arr = np.asarray(pts[:n], dtype=float)
    return arr


def save_fig(fig: plt.Figure, outdir: Path, name: str, *, dpi: int = 300) -> None:
    """Save a figure as JPG under outdir with a standard name (300 DPI)."""
    outdir.mkdir(parents=True, exist_ok=True)
    jpg = outdir / f"{name}.jpg"
    fig.savefig(jpg, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure: {jpg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="number of points")
    ap.add_argument("--angles", type=int, default=360, help="number of angles")
    ap.add_argument("--repeats", type=int, default=1000, help="trials per m")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    ap.add_argument("--outdir", type=Path, default=Path("results"), help="output dir")
    args = ap.parse_args()

    print(
        (
            f"Config: n={args.n}, angles={args.angles}, repeats={args.repeats}, "
            f"seed={args.seed}, outdir={args.outdir}"
        )
    )

    data_dir = args.outdir
    plots_dir = args.outdir / "plots"
    args.outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    t0 = time.perf_counter()
    points = sample_unit_disk_rejection(args.n, rng)
    t1 = time.perf_counter()
    r = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    print(
        (
            f"Sampled {args.n} points uniformly from unit disk in {t1 - t0:.3f}s. "
            f"points.shape={points.shape}, "
            f"r[min,mean,max]=[{r.min():.3f},{r.mean():.3f},{r.max():.3f}]"
        )
    )

    labels = np.zeros(args.n, dtype=int)
    labels[: args.n // 2] = 1
    t2 = time.perf_counter()
    angles, _ = polar_gini_curve(points, labels, num_angles=args.angles)
    t3 = time.perf_counter()
    first5 = [round(x, 2) for x in np.degrees(angles[:5])]
    print(
        (
            f"Computed base angles for PGC in {t3 - t2:.3f}s: count={len(angles)}, "
            f"first5(deg)={first5}"
        )
    )

    rows: list[tuple[int, int, float]] = []

    block = max(2, args.repeats // 10)
    for m in PERCENTAGES:
        k = int(round(args.n * m / 100))
        print(f"Processing percentage: {m}% (k={k})")
        tm0 = time.perf_counter()
        example_saved = False
        for i in range(args.repeats):
            fg_idx = rng.choice(args.n, size=k, replace=False)
            labels = np.zeros(args.n, dtype=int)
            labels[fg_idx] = 1
            _, curves = polar_gini_curve(points, labels, num_angles=args.angles)
            bg_curve, fg_curve = curves[0], curves[1]
            val = rmsd(bg_curve, fg_curve)
            rows.append((int(m), int(i), float(val)))

            if not example_saved and i == 0:
                ex = pd.DataFrame(
                    {
                        "theta_deg": np.degrees(angles),
                        "pgc_back": bg_curve,
                        "pgc_fore": fg_curve,
                    }
                )
                csv_ex = data_dir / f"pgc_sim1_example_curves_m{m}.csv"
                ex.to_csv(csv_ex, index=False)
                print(f"  Saved example curves CSV for m={m}%: {csv_ex}")

                fig2 = plt.figure(figsize=(10, 4.5))
                ax_sc = fig2.add_subplot(1, 2, 1)
                ax_pg = fig2.add_subplot(1, 2, 2, projection="polar")

                try:
                    cmap = plt.colormaps.get_cmap("tab10")  # type: ignore[attr-defined]
                except AttributeError:
                    cmap = plt.cm.get_cmap("tab10")  # type: ignore[attr-defined]
                color_bg = cmap(0)
                color_fg = cmap(1)

                mask_bg = labels == 0
                mask_fg = ~mask_bg
                sc_bg = ax_sc.scatter(
                    points[mask_bg, 0],
                    points[mask_bg, 1],
                    s=12,
                    c=[color_bg],
                    alpha=0.7,
                    linewidths=0.0,
                    label="Background",
                )
                sc_fg = ax_sc.scatter(
                    points[mask_fg, 0],
                    points[mask_fg, 1],
                    s=12,
                    c=[color_fg],
                    alpha=0.7,
                    linewidths=0.0,
                    label="Foreground",
                )
                if points.shape[0] > 50000:
                    sc_bg.set_rasterized(True)
                    sc_fg.set_rasterized(True)

                ax_sc.set_xlabel("X coordinate")
                ax_sc.set_ylabel("Y coordinate")
                ax_sc.set_title("Point cloud")
                ax_sc.grid(True, alpha=0.3)
                ax_sc.set_aspect("equal", adjustable="box")
                ax_sc.legend(loc="best")

                closed_angles = np.concatenate([angles, [2 * np.pi]])
                closed_bg = np.concatenate([bg_curve, [bg_curve[0]]])
                closed_fg = np.concatenate([fg_curve, [fg_curve[0]]])
                ax_pg.plot(
                    closed_angles,
                    closed_bg,
                    color=color_bg,
                    linewidth=2,
                    label="Background",
                )
                ax_pg.plot(
                    closed_angles,
                    closed_fg,
                    color=color_fg,
                    linewidth=2,
                    label="Foreground",
                )

                ax_pg.set_xlabel("Angle (radians)")
                ax_pg.set_ylabel("Gini coefficient", labelpad=22)
                ax_pg.set_title(f"Polar Gini Curves (m={m}%)", pad=18)
                ax_pg.set_theta_direction(-1)  # type: ignore[attr-defined]
                ax_pg.set_theta_zero_location("E")  # type: ignore[attr-defined]
                max_g = float(max(bg_curve.max(), fg_curve.max()))
                min_g = float(min(bg_curve.min(), fg_curve.min()))
                ax_pg.set_ylim(min_g * 0.95, max_g * 1.05)
                ax_pg.grid(True, alpha=0.3)
                ax_pg.legend(loc="upper left", bbox_to_anchor=(0.1, 1.1))

                fig2.tight_layout()
                save_fig(fig2, plots_dir, f"pgc_sim1_example_curves_m{m}")
                plt.close(fig2)
                example_saved = True

            if (i % block) == 0:
                print(
                    (
                        f"  m={m}% trial={i}/{args.repeats} RMSD={val:.6f} "
                        f"bg[min,max]=[{bg_curve.min():.4f},{bg_curve.max():.4f}] "
                        f"fg[min,max]=[{fg_curve.min():.4f},{fg_curve.max():.4f}]"
                    )
                )

        tm1 = time.perf_counter()
        last_block = rows[-args.repeats :]
        vals = np.fromiter((v[2] for v in last_block), dtype=float, count=args.repeats)
        print(
            (
                f"Completed m={m}% in {tm1 - tm0:.3f}s: "
                f"RMSD[min,mean,max]=[{vals.min():.6f},{vals.mean():.6f},{vals.max():.6f}]"
            )
        )

    df = pd.DataFrame(rows, columns=["m", "trial", "rmsd"])
    csv_trials = data_dir / "pgc_sim1_rmsd_by_m.csv"
    df.to_csv(csv_trials, index=False)
    print(f"Saved per-trial RMSD data to CSV: {csv_trials} (rows={len(df)})")

    t4 = time.perf_counter()
    summary = (
        df.groupby("m")["rmsd"]
        .agg(
            mean_rmsd="mean",
            sd_rmsd=lambda x: x.std(ddof=1),
            q025=lambda x: x.quantile(0.025),
            q975=lambda x: x.quantile(0.975),
            n_trials_effective="count",
        )
        .reset_index()
    )
    t5 = time.perf_counter()
    csv_summary = data_dir / "pgc_sim1_rmsd_summary.csv"
    summary.to_csv(csv_summary, index=False)
    print(
        (
            f"Saved summary statistics to CSV: {csv_summary} in {t5 - t4:.3f}s.\n"
            f"Summary head:\n{summary.head().to_string(index=False)}"
        )
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary["m"], summary["mean_rmsd"], label="Mean RMSD")
    ax.fill_between(
        summary["m"], summary["q025"], summary["q975"], alpha=0.2, label="2.5–97.5%"
    )
    ax.set_xlabel("Expressing percentage (%)")
    ax.set_ylabel("RMSD")
    ax.set_title("RMSD between foreground and background PGCs")
    ax.legend()
    fig.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_fig(fig, plots_dir, "pgc_sim1_rmsd_summary")
    plt.close(fig)

    try:
        fig3, ax3 = plt.subplots(figsize=(9, 5))
        sns.violinplot(data=df, x="m", y="rmsd", ax=ax3, inner="quartile", cut=0)
        ax3.set_xlabel("Expressing percentage (%)")
        ax3.set_ylabel("RMSD")
        ax3.set_title("RMSD distribution by expressing %")
        fig3.tight_layout()
        save_fig(fig3, plots_dir, "pgc_sim1_rmsd_violin")
        plt.close(fig3)
    except Exception as e:
        print(f"Failed to create violin plot: {e}")

    print("m%\tmean\tstd\t2.5%\t97.5%\tn")
    for _, row in summary.iterrows():
        print(
            f"{int(row['m']):2d}\t{row['mean_rmsd']:.6f}\t{row['sd_rmsd']:.6f}\t{row['q025']:.6f}\t{row['q975']:.6f}\t{int(row['n_trials_effective'])}"
        )


if __name__ == "__main__":
    main()
