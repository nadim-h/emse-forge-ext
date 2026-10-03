"""RQ1: perplexity across CodeHealth.

    python -m pipeline.rq1_perplexity

Prints every table and number; saves `figures/perplexity_by_ch.pdf` and `ppl_controls.pdf`.
"""
from __future__ import annotations

import zlib

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from pipeline import data
from pipeline.outcomes import LANG_NAMES, MODELS, RC, STYLE, print_table
from pipeline.data import ROBUST_Z, filter_outliers, load_ppl
from pipeline.stats import (
    HEALTHY,
    N_BOOT,
    QUARTILES,
    band_sums,
    bucket_bootstrap,
    bucket_partial_rhos,
    holm,
    stars,
    summarize,
    with_bucket,
    within,
)

FIG = data.FIGURES_DIR / "perplexity_by_ch.pdf"
CONTROL_FIG = data.FIGURES_DIR / "ppl_controls.pdf"
CONTROLS = [("None", []), ("SLOC", ["sloc"]), ("zlib", ["comp"])]


def cliff_label(d: float) -> str:
    a = abs(d)
    return "negligible" if a < 0.147 else "small" if a < 0.33 else "medium" if a < 0.474 else "large"


def contrast(d: pd.DataFrame) -> dict:
    """Healthy vs Unhealthy PPL: median difference, Cliff's delta (from the Mann-Whitney U) and p."""
    h, u = d.loc[d["ch"] >= HEALTHY, "ppl"], d.loc[d["ch"] < HEALTHY, "ppl"]
    mw = stats.mannwhitneyu(h, u)
    return dict(dmed=h.median() - u.median(), delta=2.0 * mw.statistic / (len(h) * len(u)) - 1.0, p=mw.pvalue,
                nh=len(h), nu=len(u))


def rank_sums(d: pd.DataFrame):
    """Sums and counts of each solution's PPL percentile within its bucket, per bucket and CH quartile."""
    return band_sums(d.assign(r=100 * d.groupby("bucket")["ppl"].rank(pct=True)), "r")


def mean_rank(s: np.ndarray, c: np.ndarray) -> np.ndarray:
    return s.sum(0) / c.sum(0)


def compression_ratio(src: bytes) -> float:
    return len(zlib.compress(src, 9)) / len(src)


def compressibility() -> pd.Series:
    """zlib-compressed over raw size of each source; lower means more repetitive."""
    return pd.Series({r["solution_id"]: compression_ratio(r["task"].encode())
                      for lang in data.LANGS for r in data.load_dataset(lang)})


def control_rhos(rhos: dict, rng) -> list[tuple[float, float, float]]:
    """Mean within-bucket ρ(CH, PPL) under each control set, with a 95% bucket-bootstrap interval."""
    out = []
    for lab, _ in CONTROLS:
        r = np.array(rhos[lab])
        boot = r[rng.integers(0, len(r), (N_BOOT, len(r)))].mean(axis=1)
        out.append((r.mean(), *np.percentile(boot, [2.5, 97.5])))
    return out


def control_figure(res: dict) -> None:
    with mpl.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(372 / 72, 2.1), sharex=True, sharey=True, layout="constrained")
        for ax, (lang, title) in zip(axes, LANG_NAMES.items()):
            ax.grid(True, axis="x", color="0.9", lw=0.5)
            ax.set_axisbelow(True)
            ax.axvline(0, color="0.4", lw=0.7, zorder=1)
            for k, (m, name) in enumerate(MODELS.items()):
                color, marker, _ = STYLE[m]
                for i, (b, lo, hi) in enumerate(res[m, lang]):
                    ax.errorbar(b, i + (k - 1) * 0.2, xerr=[[b - lo], [hi - b]], color=color, marker=marker, ms=3.5,
                                lw=0, elinewidth=0.9, capsize=1.5, label=name if i == 0 else None)
            ax.set_title(title)
            ax.set_yticks(range(len(CONTROLS)), [lab for lab, _ in CONTROLS])
            ax.set_ylim(len(CONTROLS) - 0.5, -0.5)
        axes[0].set_ylabel("Held fixed")
        fig.supxlabel("Mean within-bucket $\\rho$(CH, PPL)", fontsize=7.5)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=3, frameon=False)
        fig.savefig(CONTROL_FIG, metadata={"CreationDate": None})
        plt.close(fig)
    print(f"-> {CONTROL_FIG}")


def figure(kept: dict) -> None:
    rng = np.random.default_rng(20260926)
    with mpl.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(372 / 72, 2.3), sharey=True, layout="constrained")
        for ax, (lang, title) in zip(axes, LANG_NAMES.items()):
            ax.grid(True, color="0.9", lw=0.5)
            ax.set_axisbelow(True)
            ax.axhline(50, color="0.6", lw=0.6, zorder=0)
            for k, (m, name) in enumerate(MODELS.items()):
                color, marker, ls = STYLE[m]
                point, lo, hi = bucket_bootstrap(mean_rank, *rank_sums(kept[m][kept[m]["lang"] == lang]), rng)
                ax.errorbar(np.arange(4) + (k - 1) * 0.09, point, yerr=[lo, hi], color=color, marker=marker,
                            ls=ls, lw=1.1, ms=3.5, capsize=1.5, elinewidth=0.7, label=name)
            ax.set_title(title)
            ax.set_xticks(range(4), QUARTILES)
            ax.set_xlabel("CH quartile")
        axes[0].set_ylabel("Mean PPL rank within the bucket\n(percentile)")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=3, frameon=False)
        fig.savefig(FIG, metadata={"CreationDate": None})
        plt.close(fig)
    print(f"-> {FIG}")


def main() -> None:
    raw = {m: with_bucket(load_ppl(m)) for m in MODELS}
    kept = {m: filter_outliers(d) for m, d in raw.items()}
    print("RQ1: perplexity across CodeHealth")
    print(f"PPL upper-tail filter: robust z > {ROBUST_Z:g} per language. Healthy: CH >= {HEALTHY:g}.")

    rows = []
    for m in MODELS:
        r = raw[m]
        rows.append([MODELS[m], "all", f"{len(r):,}", f"{100 * (1 - len(kept[m]) / len(r)):.1f}%", "", ""])
        for lang, g in r.groupby("lang"):
            k = kept[m][kept[m]["lang"] == lang]
            w_raw, w_kept = stats.shapiro(g["ppl"]), stats.shapiro(k["ppl"])
            rows.append([MODELS[m], lang, f"{len(g):,}", f"{100 * (1 - len(k) / len(g)):.1f}%",
                         f"{w_raw.statistic:.3f} ({w_raw.pvalue:.2g})", f"{w_kept.statistic:.3f} ({w_kept.pvalue:.2g})"])
    print_table("Share of solutions removed by the filter, and Shapiro-Wilk W (p) of PPL before and after it",
                ["model", "lang", "n", "removed", "W raw (p)", "W filtered (p)"], rows)

    # PPL levels differ between languages, so every cell is one model in one language.
    comp = compressibility()
    cells = [(m, lang, g.assign(comp=g["solution_id"].map(comp))) for m in MODELS for lang, g in kept[m].groupby("lang")]
    c = [contrast(g) for _, _, g in cells]
    w = [within(g, "ch", "ppl") for _, _, g in cells]
    partial = [{lab: bucket_partial_rhos(g, "ch", "ppl", zs) for lab, zs in CONTROLS} for _, _, g in cells]
    wp = [summarize(r["SLOC"]) for r in partial]
    p_c, p_w, p_wp = (holm([x[key] for x in xs]) for xs, key in ((c, "p"), (w, "sign_p"), (wp, "sign_p")))
    rows = [[MODELS[m], lang, ci["nh"], ci["nu"], f"{ci['dmed']:+.3f}", f"{ci['delta']:+.3f}",
             cliff_label(ci["delta"]), f"{pc:.2g}{stars(pc)}", wi["k"],
             f"{wi['mean_rho']:+.3f}", f"{100 * wi['pos']:.0f}{stars(pw)}",
             f"{wpi['mean_rho']:+.3f}", f"{100 * wpi['pos']:.0f}{stars(pp)}"]
            for (m, lang, _), ci, wi, wpi, pc, pw, pp in zip(cells, c, w, wp, p_c, p_w, p_wp)]
    print_table("PPL of Healthy vs Unhealthy solutions (Mann-Whitney), and within-bucket rho(CH, PPL) without and "
                "with SLOC partialed out (tab:rq1; Holm over the six cells, stars on pos % are the sign test)",
                ["model", "lang", "n healthy", "n unhealthy", "median diff", "Cliff's delta", "size", "p",
                 "buckets", "rho", "pos %", "rho | SLOC", "pos % | SLOC"], rows)

    red = [(within(g, "ch", "comp"), within(g, "comp", "ppl"), summarize(r["zlib"]))
           for (_, _, g), r in zip(cells, partial)]
    p_red = holm([r[2]["sign_p"] for r in red])
    rows = [[MODELS[m], lang, f"{a['mean_rho']:+.3f}", f"{b['mean_rho']:+.3f}", f"{c['mean_rho']:+.3f}{stars(p)}",
             f"{100 * c['pos']:.0f}"] for (m, lang, _), (a, b, c), p in zip(cells, red, p_red)]
    print_table("Within-bucket rho with the compression ratio (zlib, level 9) of the source, and rho(CH, PPL) "
                "with the ratio partialed out (Holm over the six cells)",
                ["model", "lang", "rho(CH, ratio)", "rho(ratio, PPL)", "rho(CH, PPL) | ratio", "pos %"], rows)

    rng = np.random.default_rng(20260927)
    res = {(m, lang): control_rhos(r, rng) for (m, lang, _), r in zip(cells, partial)}
    rows = [[MODELS[m], lang, *(f"{b:+.3f} [{lo:+.3f}, {hi:+.3f}]" for b, lo, hi in res[m, lang])]
            for m, lang, _ in cells]
    print_table("Mean within-bucket rho(CH, PPL) with each control partialed out, and its 95% bucket-bootstrap "
                "interval (fig:ppl-controls)", ["model", "lang", *(lab for lab, _ in CONTROLS)], rows)

    rows = [[MODELS[m], lang, *(f"{v:.1f}" for v in mean_rank(*rank_sums(g)))] for m, lang, g in cells]
    print_table("Mean within-bucket PPL percentile by CH quartile, 50 being the bucket median (fig:ppl)",
                ["model", "lang", *QUARTILES], rows)
    print()
    figure(kept)
    control_figure(res)


if __name__ == "__main__":
    main()
