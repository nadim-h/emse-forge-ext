"""RQ3: CodeHealth against problem difficulty, program size and perplexity as predictors of task outcomes.

    python -m pipeline.rq3_predictors

Prints every table and number; saves `figures/rq3_coefficients.pdf`, `difficulty_by_problem.pdf`
and `tree_importance.pdf`.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from pipeline import data
from pipeline.outcomes import LANG_NAMES, MODELS, RC, STYLE, load_task, print_table, with_rating
from pipeline.data import filter_outliers, load_ppl
from pipeline.stats import ch_bands, fit_clustered, holm, rating_quartiles, stars

OUTCOMES = [("A", "mutation_score", "Mutation score"), ("A", "branch_rate", "Branch coverage"),
            ("B", "fixed", "Repair success"), ("C", "preserved", "Behavior preserved")]
# The conference paper's trees also split on SLOC; here SLOC is only a regression control (tab:joint).
TREE_FEATURES = [("ch", "CH"), ("ppl", "PPL"), ("rating", "Difficulty")]
TREE_COLUMNS = [f for f, _ in TREE_FEATURES]
FIG = data.FIGURES_DIR / "rq3_coefficients.pdf"
DIFF_FIG = data.FIGURES_DIR / "difficulty_by_problem.pdf"
TREE_FIG = data.FIGURES_DIR / "tree_importance.pdf"
# fig:joint rows: what is held fixed besides the bucket.
CH_ROWS = [("ch_raw", "No controls"), ("ch_ppl", "PPL fixed"), ("ch_sloc", "Size fixed")]


def frame(m: str, task: str, ppl: pd.DataFrame) -> pd.DataFrame:
    return with_rating(load_task(m, task).merge(ppl[["solution_id", "ppl"]], on="solution_id", how="left"))


def bucket_means(d: pd.DataFrame, col: str) -> pd.DataFrame:
    """Each bucket's rating `r`, mean outcome `y` and language."""
    return d.groupby("bucket").agg(r=("rating", "first"), y=(col, "mean"), lang=("lang", "first"))


def rating_quartile_counts(d: pd.DataFrame) -> None:
    """Buckets per rating quartile and language."""
    b = d.groupby("bucket")[["rating", "lang"]].first()
    quart = pd.crosstab(rating_quartiles(b["rating"]), b["lang"])
    print("Buckets per rating quartile: "
          + "; ".join(f"Q{i + 1} {r['cpp']} C++ / {r['java']} Java" for i, r in quart.iterrows()))


def difficulty_section(frames: dict) -> None:
    """Difficulty against the bucket-mean outcome, over all buckets and per language, and its quartile gap next to CH's."""
    rows = []
    for m in MODELS:
        for t, c, lab in OUTCOMES:
            d = frames[m, t].dropna(subset=[c])
            b = bucket_means(d, c)
            by_rating = b.groupby(rating_quartiles(b["r"]))["y"].mean()
            by_ch = d.groupby(ch_bands(d["ch"]), observed=True)[c].mean()
            by_lang = [stats.spearmanr(g["r"], g["y"]) for _, g in b.groupby("lang")]
            rows.append([MODELS[m], lab, *stats.spearmanr(b["r"], b["y"]), *[x for r in by_lang for x in r],
                         by_rating.iloc[0] - by_rating.iloc[-1], by_ch.iloc[-1] - by_ch.iloc[0]])
    adj = holm([r[3] for r in rows])
    adj_lang = holm([r[k] for r in rows for k in (5, 7)]).reshape(-1, 2)
    print_table("Spearman rho over buckets of the rating against the bucket-mean outcome, overall (Holm over the "
                f"table) and per language (Holm over the {adj_lang.size} cells), and the outcome gap between the "
                "easiest and hardest rating quartile next to the gap between CH quartiles (tab:difficulty)",
                ["model", "outcome", "rho", "p", "rho C++", "p C++", "rho Java", "p Java",
                 "difficulty Q1−Q4", "CH Q4−Q1", "ratio"],
                [[n, lab, f"{r:+.3f}", f"{p:.2g}{stars(p)}", f"{rc:+.3f}", f"{pc:.2g}{stars(pc)}",
                  f"{rj:+.3f}", f"{pj:.2g}{stars(pj)}", f"{gd:+.3f}", f"{gc:+.3f}", f"{gd / gc:.1f}"]
                 for (n, lab, r, _, rc, _, rj, _, gd, gc), p, (pc, pj) in zip(rows, adj, adj_lang)])


def difficulty_figure(frames: dict) -> None:
    """fig:difficulty: one point per bucket, rating against mean outcome, one OLS line per model and language."""
    with mpl.rc_context(RC):
        fig, axes = plt.subplots(2, 2, figsize=(372 / 72, 3.7), sharex=True, layout="constrained")
        for ax, (t, c, lab) in zip(axes.flat, OUTCOMES):
            ax.grid(True, color="0.92", lw=0.5)
            ax.set_axisbelow(True)
            for m in MODELS:
                color, marker, _ = STYLE[m]
                b = bucket_means(frames[m, t].dropna(subset=[c]), c)
                for lang, g in b.groupby("lang"):
                    filled = lang == "cpp"
                    ax.scatter(g["r"], g["y"], s=9, marker=marker, lw=0 if filled else 0.6, alpha=0.8,
                               color=color if filled else "none", edgecolors="none" if filled else color)
                    slope, icpt = np.polyfit(g["r"], g["y"], 1)
                    # The line stops where it would leave [0, 1], the range of every outcome.
                    xs = np.linspace(g["r"].min(), g["r"].max(), 200)
                    ys = icpt + slope * xs
                    keep = (ys >= 0) & (ys <= 1)
                    ax.plot(xs[keep], ys[keep], color=color, lw=0.9, ls="-" if filled else "--")
            ax.set_title(f"Task {t}: {lab.lower()}")
        fig.supxlabel("Problem difficulty rating", fontsize=7.5)
        fig.supylabel("Mean outcome over the bucket's solutions", fontsize=7.5)
        handles = [Line2D([], [], color=STYLE[m][0], marker=STYLE[m][1], ls="none", ms=4, label=name)
                   for m, name in MODELS.items()]
        handles += [Line2D([], [], color="0.3", marker="o", ls="-", lw=0.9, ms=4, label="C++"),
                    Line2D([], [], color="0.3", marker="o", mfc="none", ls="--", lw=0.9, ms=4, label="Java")]
        fig.legend(handles=handles, loc="outside upper center", ncol=5, frameon=False)
        fig.savefig(DIFF_FIG, metadata={"CreationDate": None})
        plt.close(fig)
    print(f"-> {DIFF_FIG}")


def joint_fits(frames: dict) -> list[dict]:
    """Per-SD bucket fixed-effects coefficients of CH under each control set, and of the PPL rank."""
    out = []
    for m in MODELS:
        for t, c, lab in OUTCOMES:
            d = frames[m, t].dropna(subset=[c, "ppl"])
            d = d.assign(y=d[c], lsloc=np.log(d["sloc"]),
                         rppl=d.groupby("bucket")["ppl"].rank(pct=True))
            for v in ("ch", "lsloc", "rppl"):
                d[f"z_{v}"] = (d[v] - d[v].mean()) / d[v].std()
            raw = fit_clustered(d, "y ~ z_ch + C(bucket)")
            sloc = fit_clustered(d, "y ~ z_ch + z_lsloc + C(bucket)")
            with_ppl = fit_clustered(d, "y ~ z_ch + z_rppl + C(bucket)")
            joint = fit_clustered(d, "y ~ z_ch + z_lsloc + z_rppl + C(bucket)")
            for fit, key, v in ((raw, "ch_raw", "z_ch"), (sloc, "ch_sloc", "z_ch"), (with_ppl, "ch_ppl", "z_ch"),
                                (joint, "ch", "z_ch"), (joint, "ppl", "z_rppl")):
                lo, hi = fit.conf_int().loc[v]
                out.append(dict(model=m, task=t, outcome=lab, term=key, b=fit.params[v], lo=lo, hi=hi,
                                p=fit.pvalues[v]))
    return out


def joint_table(fits: list[dict]) -> None:
    cell = {(f["model"], f["outcome"], f["term"]): f"{f['b']:+.4f}{stars(f['p'])}" for f in fits}
    b = {(f["model"], f["outcome"], f["term"]): f["b"] for f in fits}
    rows = [[MODELS[m], f"{t} — {lab}", *(cell[m, lab, k] for k in ("ch_raw", "ch_sloc", "ch_ppl", "ch", "ppl"))]
            for m in MODELS for t, _, lab in OUTCOMES]
    print_table("Coefficient per +1 SD with bucket fixed effects, clustered SEs and unadjusted p, of CH under each "
                "control set and of the PPL rank (tab:joint; the first three columns are in rq3_coefficients.pdf)",
                ["model", "outcome", "CH", "CH | SLOC", "CH | PPL", "CH | SLOC, PPL", "PPL rank | SLOC, CH"], rows)

    def ci(f):
        return [MODELS[f["model"]], f"{f['task']} — {f['outcome']}", f"{100 * f['lo']:+.1f}", f"{100 * f['hi']:+.1f}"]

    print_table("95% CI of CH with SLOC held fixed, Tasks B and C, percentage points per SD", ["model", "outcome", "lo", "hi"],
                [ci(f) for f in fits if f["term"] == "ch_sloc" and f["task"] in ("B", "C")])
    print_table("95% CI of CH with PPL held fixed, percentage points per SD", ["model", "outcome", "lo", "hi"],
                [ci(f) for f in fits if f["term"] == "ch_ppl"])
    print_table("Share of the uncontrolled CH coefficient that remains with SLOC held fixed, Task A",
                ["model", "outcome", "share"],
                [[MODELS[m], lab, f"{100 * b[m, lab, 'ch_sloc'] / b[m, lab, 'ch_raw']:.0f}%"]
                 for m in MODELS for t, _, lab in OUTCOMES if t == "A"])


def figure(fits: list[dict]) -> None:
    """fig:joint: the CH coefficient per control set, pp per +1 SD with 95% CIs."""
    f = pd.DataFrame(fits).assign(b=lambda x: 100 * x.b, lo=lambda x: 100 * x.lo, hi=lambda x: 100 * x.hi)
    with mpl.rc_context(RC):
        fig, axes = plt.subplots(2, 2, figsize=(372 / 72, 3.1), sharex=True, sharey=True, layout="constrained")
        for ax, (t, _, lab) in zip(axes.flat, OUTCOMES):
            ax.grid(True, axis="x", color="0.9", lw=0.5)
            ax.set_axisbelow(True)
            ax.axvline(0, color="0.4", lw=0.7, zorder=1)
            for i, (term, _) in enumerate(CH_ROWS):
                for k, m in enumerate(MODELS):
                    color, marker, _ = STYLE[m]
                    r = f[(f.model == m) & (f.outcome == lab) & (f.term == term)].iloc[0]
                    ax.errorbar(r.b, i + (k - 1) * 0.22, xerr=[[r.b - r.lo], [r.hi - r.b]], color=color,
                                marker=marker, ms=3.2, lw=0, elinewidth=0.9, capsize=1.3)
            ax.set_title(f"Task {t}: {lab.lower()}")
            ax.set_yticks(range(len(CH_ROWS)), [name for _, name in CH_ROWS])
            ax.set_ylim(len(CH_ROWS) - 0.5, -0.5)
        fig.supxlabel("Change in outcome per +1 SD of CH (percentage points)", fontsize=7.5)
        handles = [Line2D([], [], color=STYLE[m][0], marker=STYLE[m][1], ms=3.5, ls="none", label=name)
                   for m, name in MODELS.items()]
        fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False)
        fig.savefig(FIG, metadata={"CreationDate": None})
        plt.close(fig)
    print(f"-> {FIG}")


def tree(d: pd.DataFrame, col: str) -> dict:
    """Depth-3 classification tree for failure (1 - col), as in the conference paper."""
    x = d.dropna(subset=[col])
    y = 1 - x[col].astype(int)
    X = x[TREE_COLUMNS].to_numpy(float)
    mk = lambda: DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, class_weight="balanced", random_state=0)
    clf = mk().fit(X, y)
    cv = cross_val_predict(mk(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")
    return dict(n=len(x), fail=y.mean(), auc_cv=roc_auc_score(y, cv[:, 1]), imp=clf.feature_importances_,
                root=TREE_FEATURES[clf.tree_.feature[0]][1])


def regression_tree(d: pd.DataFrame, col: str) -> dict:
    """The same tree for a continuous outcome (Task A's mutation score)."""
    x = d.dropna(subset=[col])
    X, y = x[TREE_COLUMNS].to_numpy(float), x[col].to_numpy(float)
    mk = lambda: DecisionTreeRegressor(max_depth=3, min_samples_leaf=50, random_state=0)
    reg = mk().fit(X, y)
    cv = cross_val_predict(mk(), X, y, cv=KFold(5, shuffle=True, random_state=0))
    return dict(n=len(x), r2_cv=r2_score(y, cv), imp=reg.feature_importances_,
                root=TREE_FEATURES[reg.tree_.feature[0]][1])


# Task, outcome, title, and whether the tree classifies pass/fail.
TREE_TASKS = [("A", "mutation_score", "Task A: test generation", False), ("B", "fixed", "Task B: bug fixing", True),
              ("C", "preserved", "Task C: refactoring", True)]
TREE_LANGS = ("cpp", "java", "all")


def fit_trees(filtered: dict) -> dict:
    """Every tree of tab:trees-lang, keyed by model, task and language."""
    out = {}
    for m in MODELS:
        for t, c, _, binary in TREE_TASKS:
            for lang in TREE_LANGS:
                d = filtered[m, t]
                d = d if lang == "all" else d[d["lang"] == lang]
                out[m, t, lang] = tree(d, c) if binary else regression_tree(d, c)
    return out


def tree_roots(trees: dict) -> None:
    """Root splits of the per-language trees drawn in fig:trees, counted per task."""
    rows = []
    for t, _, title, _ in TREE_TASKS:
        rs = [trees[m, t, lang] for m in MODELS for lang in ("cpp", "java")]
        rows.append([title, len(rs), ", ".join(f"{k} {sum(r['root'] == k for r in rs)}" for _, k in TREE_FEATURES),
                     f"{min(r['imp'][0] for r in rs):.3f}–{max(r['imp'][0] for r in rs):.3f}"])
    print_table("Root splits of the per-language trees, C++ and Java over three models (fig:trees)",
                ["task", "trees", "root", "CH importance"], rows)


def tree_figure(trees: dict) -> None:
    """fig:trees: feature importance, one stacked bar per model and language."""
    color = {"CH": "#0072B2", "PPL": "#E69F00", "Difficulty": "0.6"}
    with mpl.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(372 / 72, 2.1), sharey=True, layout="constrained")
        for ax, (t, _, title, _) in zip(axes, TREE_TASKS):
            ticks, labels, y = [], [], 0.0
            for m, name in MODELS.items():
                # Per language only; the pooled trees are in tab:trees-lang.
                for lang, lang_name in LANG_NAMES.items():
                    r = trees[m, t, lang]
                    ax.barh(y, r["imp"], left=np.r_[0, np.cumsum(r["imp"])[:-1]], height=0.72, edgecolor="white",
                            lw=0.4, color=[color[lab] for _, lab in TREE_FEATURES])
                    ax.plot(1.06, y, marker="o", ms=3.2, color=color[r["root"]], clip_on=False)
                    ticks.append(y)
                    labels.append(f"{name}, {lang_name}")
                    y += 1
                y += 0.5
            ax.set_title(title)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        axes[0].set_yticks(ticks, labels)
        axes[0].invert_yaxis()
        fig.supxlabel("Feature importance (dot: feature at the root split)", fontsize=7.5)
        fig.legend(handles=[Patch(color=color[k], label=k) for k in ("CH", "PPL", "Difficulty")],
                   loc="outside upper center", ncol=3, frameon=False)
        fig.savefig(TREE_FIG, metadata={"CreationDate": None})
        plt.close(fig)
    print(f"-> {TREE_FIG}")


def trees_table(trees: dict) -> None:
    """tab:trees-lang: fit is the 5-fold CV AUC (B, C) or R^2 (A)."""
    rows = []
    for m in MODELS:
        for t, _, _, binary in TREE_TASKS:
            for lang in TREE_LANGS:
                r = trees[m, t, lang]
                rows.append([MODELS[m], t, lang, f"{r['n']:,}", f"{100 * r['fail']:.1f}" if binary else "--",
                             f"{r['auc_cv'] if binary else r['r2_cv']:.3f}", *(f"{v:.3f}" for v in r["imp"]), r["root"]])
    print_table("Trees pooled over buckets, PPL outliers filtered: fit is the 5-fold CV AUC of failure for "
                "Tasks B and C and the 5-fold CV R^2 of the mutation score for Task A (tab:trees-lang)",
                ["model", "task", "lang", "n", "% fail", "fit", *(lab for _, lab in TREE_FEATURES), "root"], rows)


def main() -> None:
    ppl = {m: load_ppl(m) for m in MODELS}
    tasks = {t for t, _, _ in OUTCOMES}
    frames = {(m, t): frame(m, t, ppl[m]) for m in MODELS for t in tasks}
    filtered = {(m, t): frame(m, t, filter_outliers(ppl[m])).dropna(subset=["ppl"])
                for m in MODELS for t in tasks}
    fits = joint_fits(frames)
    trees = fit_trees(filtered)
    print("RQ3: CodeHealth against difficulty, size and perplexity")
    rating_quartile_counts(frames[next(iter(MODELS)), "A"])
    difficulty_section(frames)
    joint_table(fits)
    trees_table(trees)
    tree_roots(trees)
    print()
    figure(fits)
    difficulty_figure(frames)
    tree_figure(trees)


if __name__ == "__main__":
    main()
