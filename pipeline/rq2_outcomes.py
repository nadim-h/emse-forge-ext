"""The corpus and RQ2: task outcomes across CodeHealth.

    python -m pipeline.rq2_outcomes

Prints every table and number; saves `figures/outcome_by_ch.pdf`, `rho_by_bucket.pdf` and `ch_by_bucket.pdf`.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from pipeline import data
from pipeline.outcomes import LANG_NAMES, MODELS, RC, STYLE, load_corpus, load_task, print_table
from pipeline.stats import (
    CH_QUARTILE_EDGES,
    HEALTHY,
    QUARTILES,
    band_sums,
    bucket_bootstrap,
    bucket_rhos,
    ch_bands,
    fixed_effects,
    holm,
    pooled_spearman,
    rating_quartiles,
    stars,
    within,
)

# tab:tasks; one task's tests across the three models form one Holm family.
OUTCOMES = {
    "A": [("validity_rate", "Test validity"), ("mutation_score", "Mutation score"),
          ("branch_rate", "Branch coverage")],
    "B": [("fixed", "Repair success")],
    "C": [("preserved", "Behavior preserved")],
}
BAND_OUTCOMES = [("A", "mutation_score", "Mutation score (A)"), ("A", "branch_rate", "Branch coverage (A)"),
                 ("A", "line_rate", "Line coverage (A)"), ("A", "validity_rate", "Test validity (A)"),
                 ("B", "fixed", "Repair success (B)"), ("C", "preserved", "Behavior preserved (C)"),
                 ("C", "effective", "Effective refactoring (C)"), ("C", "ch_delta", "Mean ΔCH (C)")]
# tab:operator: Task A kill rate at CH >= GAP_HIGH minus at CH < GAP_LOW.
GAP_HIGH, GAP_LOW = 8.5, 7.0
PRIMARY = {"mutation_score", "fixed", "preserved"}
# Mutants and seeded faults are counted from the first model's results.
FIRST = next(iter(MODELS))
FIG = data.FIGURES_DIR / "outcome_by_ch.pdf"
RHO_FIG = data.FIGURES_DIR / "rho_by_bucket.pdf"
BUCKET_FIG = data.FIGURES_DIR / "ch_by_bucket.pdf"
LANG_STYLE = {"cpp": ("C++", "#0072B2"), "java": ("Java", "#D55E00")}
RHO_OUTCOMES = [("A", "validity_rate", "Task A\ntest validity"), ("A", "mutation_score", "Task A\nmutation score"),
                ("A", "branch_rate", "Task A\nbranch coverage"), ("B", "fixed", "Task B\nrepair success"),
                ("C", "preserved", "Task C\nbehavior preserved")]


def f3(x) -> str:
    return f"{x:+.3f}"


def corpus_numbers(d: pd.DataFrame) -> None:
    def col(g):
        size = g.groupby("bucket").size()
        span = g.groupby("bucket")["ch"].agg(lambda x: x.max() - x.min())
        rating = g.drop_duplicates("bucket")["rating"]
        within_var = ((g["ch"] - g.groupby("bucket")["ch"].transform("mean")) ** 2).sum()
        return [f"{len(g):,}", g["bucket"].nunique(),
                f"{size.mean():.1f} ({size.min()}–{size.max()})",
                f"{g['ch'].mean():.2f} ({g['ch'].std():.2f})", f"{g['ch'].min():.2f}-{g['ch'].max():.2f}",
                f"{g['sloc'].mean():.1f} ({g['sloc'].min()}–{g['sloc'].max()})",
                f"{rating.mean():.0f} ({rating.std():.0f})", f"{rating.min():.0f}-{rating.max():.0f}",
                f"{span.mean():.2f} ({span.std():.2f})", f"{(span >= 3).sum()} of {len(span)}",
                f"{g.groupby('bucket')['ch'].mean().std():.2f}",
                f"{100 * within_var / ((g['ch'] - g['ch'].mean()) ** 2).sum():.1f}%",
                f"{g['n_oracle'].mean():.1f} ({g['n_oracle'].min():.0f}-{g['n_oracle'].max():.0f})"]
    labels = ["Solutions", "Buckets", "Solutions/bucket, mean (min-max)", "CH, mean (sd)", "CH, min-max",
              "SLOC, mean (min–max)", "Difficulty over buckets, mean (sd)", "Difficulty, min-max",
              "CH span in a bucket, mean (sd)", "Buckets spanning >= 3 CH", "Bucket mean CH, sd",
              "CH variance within problems", "Oracle tests, mean (min-max)"]
    cols = [col(d[d["lang"] == "cpp"]), col(d[d["lang"] == "java"]), col(d)]
    print_table("Corpus by language (tab:corpus)", ["", "C++", "Java", "Total"],
                [[lab, *c] for lab, *c in zip(labels, *cols)])

    b = d.groupby("bucket").agg(ch=("ch", "mean"), rating=("rating", "first"), lang=("lang", "first"))
    rho = [["solutions", *stats.spearmanr(d["ch"], d["rating"])]]
    rho += [[f"buckets, {lab}", *stats.spearmanr(g["ch"], g["rating"])]
            for lab, g in (("all", b), ("C++", b[b.lang == "cpp"]), ("Java", b[b.lang == "java"]))]
    print_table("Spearman rho(CH, difficulty rating), over solutions and over bucket means", ["level", "rho", "p"],
                [[r, f3(x), f"{p:.2f}"] for r, x, p in rho])

    print()
    print(f"Spearman rho(CH, SLOC): {f3(stats.spearmanr(d['ch'], d['sloc'])[0])}")
    print(f"CH quartile cuts: {d['ch'].quantile([.25, .5, .75]).round(2).tolist()}")
    print(f"Solutions per CH quartile: {ch_bands(d['ch']).value_counts(sort=False).tolist()}")
    print(f"Healthy solutions (CH >= {HEALTHY:g}): {(d['ch'] >= HEALTHY).sum()}")
    print(f"Solutions below CH 7: {(d['ch'] < 7).sum()} ({100 * (d['ch'] < 7).mean():.0f}%)")
    by_q = d.assign(java=d["lang"] == "java").groupby(ch_bands(d["ch"]), observed=True)
    print("Mean rating by CH quartile: " + " / ".join(f"{v:.0f}" for v in by_q["rating"].mean()))
    print("Java share by CH quartile: " + " / ".join(f"{100 * v:.0f}%" for v in by_q["java"].mean()))
    per = pd.crosstab(d["bucket"], ch_bands(d["ch"]))
    share = per.div(per.sum(axis=1), axis=0)
    print(f"Buckets with solutions in every CH quartile: {(per > 0).all(axis=1).sum()} of {len(per)}")
    print(f"Fewest solutions of a bucket in one CH quartile: {per.min().min()}")
    print(f"Share of a bucket in one CH quartile: {100 * share.min().min():.0f}–{100 * share.max().max():.0f}%, "
          "mean by quartile " + " / ".join(f"{100 * v:.0f}%" for v in share.mean()))


def association_rows(d: pd.DataFrame, outcomes, model: str) -> list[dict]:
    return [dict(model=model, outcome=lab, n=int(d[c].notna().sum()), rate=d[c].mean(),
                 pooled=pooled_spearman(d, "ch", c), w=within(d, "ch", c), fe=fixed_effects(d, c))
            for c, lab in outcomes]


def association_table(rows: list[dict]) -> list[list]:
    sign = holm([r["w"]["sign_p"] for r in rows])
    fe = holm([r["fe"]["p"] for r in rows])
    return [[MODELS[r["model"]], r["outcome"], f"{r['n']:,}", f"{r['rate']:.3f}", f3(r["pooled"]),
             f3(r["w"]["mean_rho"]), f"{100 * r['w']['pos']:.0f}{stars(ps)}",
             f"{r['fe']['coef']:+.4f}{stars(pf) or ' --'}"] for r, ps, pf in zip(rows, sign, fe)]


def tasks_tables(frames: dict) -> None:
    for t, outs in OUTCOMES.items():
        rows = [r for m in MODELS for r in association_rows(frames[m, t], outs, m)]
        print_table(f"Task {t}: rho(CH, outcome) pooled and within buckets, and the bucket fixed-effects "
                    "coefficient (tab:tasks; Holm over the task's outcomes and models)",
                    ["model", "outcome", "n", "rate", "pooled rho", "within rho", "pos %", "FE coef"],
                    association_table(rows))


def bands_table(frames: dict) -> None:
    rows = []
    for t, c, lab in BAND_OUTCOMES:
        for m, name in MODELS.items():
            d = frames[m, t].dropna(subset=[c])
            means = d.groupby(ch_bands(d["ch"]), observed=True)[c].mean()
            fmt = "{:+.2f}" if c == "ch_delta" else "{:.3f}"
            rows.append([lab, name, *(fmt.format(v) for v in means),
                         fmt.format(means.iloc[-1] - means.iloc[0]),
                         f"{100 * (means.iloc[-1] / means.iloc[0] - 1):+.0f}%" if c in PRIMARY else ""])
    print_table("Mean outcome by CH quartile (tab:bands)", ["outcome", "model", *QUARTILES, "Q4−Q1", "relative"], rows)


def replication_table(frames: dict) -> None:
    """The conference paper's Healthy-vs-Unhealthy break-rate contrast, pooled and within buckets."""
    rows = []
    for m, name in MODELS.items():
        d = frames[m, "C"].assign(brk=lambda x: 1 - x["preserved"])
        h = d["ch"] >= HEALTHY
        bh, bu = d.loc[h, "brk"].mean(), d.loc[~h, "brk"].mean()
        diffs = np.array([g.loc[g.ch < HEALTHY, "brk"].mean() - g.loc[g.ch >= HEALTHY, "brk"].mean()
                          for _, g in d.groupby("bucket")
                          if (g.ch >= HEALTHY).any() and (g.ch < HEALTHY).any()])
        npos, n = int((diffs > 0).sum()), len(diffs)
        rows.append([name, h.sum(), (~h).sum(), f"{bh:.3f}", f"{bu:.3f}", f"{100 * (bu - bh) / bu:+.1f}%",
                     len(diffs), f3(diffs.mean()), f"{100 * npos / n:.0f}", f"{stats.binomtest(npos, n).pvalue:.2f}"])
    print_table("Task C break rate (1 - preserved) of Healthy vs Unhealthy solutions, pooled and as the mean "
                "difference within buckets that have both (tab:taskc-replication)",
                ["model", "n healthy", "n unhealthy", "healthy", "unhealthy", "RRR", "buckets", "mean diff",
                 "pos %", "sign p"], rows)


def operator_table(frames: dict) -> None:
    """Task A kill-rate gap and Task B repair success, by operator family."""
    kills = pd.DataFrame([dict(model=m, op=op, ch=ch, killed=k, total=n)
                          for m in MODELS
                          for ch, ops in frames[m, "A"].dropna(subset=["mutation_score"])[["ch", "kills"]].values
                          for op, (k, n) in ops.items()])

    def summary(k: pd.DataFrame) -> pd.Series:
        rate = lambda s: s["killed"].sum() / s["total"].sum()
        return pd.Series({"total": k["total"].sum(), "gap": rate(k[k.ch >= GAP_HIGH]) - rate(k[k.ch < GAP_LOW]),
                          "kill": rate(k)})

    by_op = pd.concat([kills.groupby(["model", "op"]).apply(summary),
                       kills.groupby("model").apply(summary).assign(op="All").set_index("op", append=True)])
    faults = pd.concat([frames[m, "B"].assign(model=m) for m in MODELS])
    repair = pd.concat([faults.groupby(["operator", "model"])["fixed"].mean().unstack(),
                        faults.groupby("model")["fixed"].mean().to_frame("All").T])
    ops = by_op.loc[FIRST].drop("All").sort_values("total", ascending=False).index.tolist() + ["All"]
    rows = [[o, f"{by_op.loc[(FIRST, o), 'total']:,.0f}", *(f3(by_op.loc[(m, o), "gap"]) for m in MODELS),
             f"{len(faults[(faults.model == FIRST) & ((faults.operator == o) | (o == 'All'))]):,}",
             *(f"{repair.loc[o, m]:.3f}" for m in MODELS)] for o in ops]
    ch_by_op = faults[faults.model == FIRST].groupby("operator")["ch"].mean()
    print()
    print("Task A kill rate over all mutants: " + ", ".join(
        f"{MODELS[m]} {by_op.loc[(m, 'All'), 'kill']:.3f}" for m in MODELS))
    print(f"Span of the mean CH across fault families: {ch_by_op.max() - ch_by_op.min():.2f}")
    print_table(f"By operator family: {MODELS[FIRST]}'s scored mutants, the Task A kill-rate gap between high and "
                "low CH per model, the seeded faults, and Task B repair success per model (tab:operator)",
                ["family", "mutants", *(f"gap {n}" for n in MODELS.values()), "faults", *MODELS.values()], rows)


def example_pair(frames: dict) -> None:
    """fig:example: in one bucket, a CH < 6 solution under 90 SLOC and a Healthy one within 10 SLOC of it, scored
    by every model, with the largest smallest-over-models mutation-score gap."""
    a = {m: frames[m, "A"].set_index("solution_id") for m in MODELS}
    best = (-1, "", "")
    for _, g in a[FIRST].dropna(subset=["mutation_score"]).groupby("bucket"):
        for i, r in g[g.ch < 6].iterrows():
            for j, s in g[g.ch >= HEALTHY].iterrows():
                if r.sloc >= 90 or abs(r.sloc - s.sloc) > 10:
                    continue
                gaps = [a[m].loc[j, "mutation_score"] - a[m].loc[i, "mutation_score"]
                        for m in MODELS if i in a[m].index and j in a[m].index]
                if len(gaps) == len(MODELS) and not np.isnan(gaps).any():
                    best = max(best, (min(gaps), i, j))
    c = {m: frames[m, "C"].set_index("solution_id") for m in MODELS}
    rows = [[sid, f"{a[m].loc[sid, 'ch']:.2f}", int(a[m].loc[sid, "sloc"]), MODELS[m],
             *(f"{a[m].loc[sid, k]:.2f}" for k in ("validity_rate", "mutation_score", "branch_rate")),
             int(c[m].loc[sid, "preserved"])] for sid in best[1:] for m in MODELS]
    print_table("Example pair of solutions (fig:example)",
                ["solution", "CH", "SLOC", "model", "test validity", "mutation score", "branch coverage", "preserved"],
                rows)


def methods_numbers(frames: dict, corpus: pd.DataFrame) -> None:
    # Mutants are the same for every model but recorded only where a model ran them: take the max.
    a = pd.concat([frames[m, "A"][["solution_id", "lang", "n_mutants", "n_mutants_compiled"]] for m in MODELS])
    mut = a.groupby(["solution_id", "lang"])[["n_mutants", "n_mutants_compiled"]].max().reset_index()
    mut = mut[mut["n_mutants"] > 0]
    fail = mut.groupby("lang")[["n_mutants", "n_mutants_compiled"]].sum()
    fail = 1 - fail["n_mutants_compiled"] / fail["n_mutants"]
    print()
    print(f"Mutants: {int(mut['n_mutants_compiled'].sum()):,} compilable of {int(mut['n_mutants'].sum()):,} "
          f"generated, over {len(mut):,} solutions")
    print("Mutants failing to compile: " + ", ".join(
        f"{LANG_NAMES[lang]} {100 * v:.1f}%" for lang, v in fail.items()))
    print(f"Compilable mutants per solution: {mut['n_mutants_compiled'].mean():.2f} "
          f"({mut['n_mutants_compiled'].sum() / len(corpus):.2f} over all {len(corpus):,})")
    rows = []
    for m, name in MODELS.items():
        d = frames[m, "A"]
        no_valid = d["n_valid"] == 0
        by_q = no_valid.groupby(ch_bands(d["ch"]), observed=True).mean()
        rows.append([name, f"{int(no_valid.sum()):,}", f"{100 * no_valid.mean():.1f}",
                     " / ".join(f"{100 * v:.1f}" for v in by_q),
                     f"{int(d['mutation_score'].notna().sum()):,}",
                     f3(within(d.dropna(subset=['mutation_score']), "ch", "n_mutants_compiled")["mean_rho"])])
    print_table("Task A solutions with no valid test, those with a mutation score, and within-bucket "
                "rho(CH, compilable mutants)",
                ["model", "no valid test", "%", "% by CH quartile", "mutation score n", "rho(CH, mutants)"], rows)
    faults = frames[FIRST, "B"].groupby("solution_id").size()
    per = corpus["solution_id"].map(faults).fillna(0).astype(int).value_counts().sort_index()
    print()
    print(f"Seeded faults: {int(faults.sum()):,}")
    print("Solutions by number of faults: " + ", ".join(
        f"{k} faults {v:,} ({100 * v / len(corpus):.1f}%)" for k, v in per.items()))
    problems = set(corpus["problem"])
    rated = pd.DataFrame([v for k, v in data.ratings().items() if k in problems])
    src = rated.drop_duplicates("canonical")["source"].str.replace("-mirror", "")
    print(f"Rating source over the {len(src)} problems: " + ", ".join(f"{k} {v}" for k, v in src.value_counts().items()))


def text_numbers(frames: dict) -> None:
    prim = {t: np.mean([within(frames[m, t], "ch", c)["mean_rho"] for m in MODELS])
            for t, c in (("A", "mutation_score"), ("B", "fixed"), ("C", "preserved"))}
    print()
    print("Mean within-bucket rho of the primary outcome over models: "
          + ", ".join(f"{t} {v:+.3f}" for t, v in prim.items()))
    print(f"A over B: {prim['A'] / prim['B']:.1f}×, A over C: {prim['A'] / prim['C']:.1f}×")


def gap_from_q1(s: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Each CH quartile's mean minus Q1's (pp), from per-bucket sums and counts."""
    return 100 * (s.sum(0) / c.sum(0) - s.sum(0)[0] / c.sum(0)[0])


def figure(frames: dict) -> None:
    panels = [("A", "mutation_score", "Task A: mutation score"), ("A", "validity_rate", "Task A: test validity"),
              ("B", "fixed", "Task B: repair success"), ("C", "preserved", "Task C: behavior preserved")]
    rng = np.random.default_rng(20260923)
    with mpl.rc_context(RC):
        fig, axes = plt.subplots(2, 2, figsize=(372 / 72, 3.6), sharex=True, sharey=True, layout="constrained")
        for ax, (t, col, title) in zip(axes.flat, panels):
            ax.grid(True, color="0.9", lw=0.5)
            ax.set_axisbelow(True)
            ax.axhline(0, color="0.6", lw=0.6, zorder=0)
            for k, (m, name) in enumerate(MODELS.items()):
                color, marker, ls = STYLE[m]
                point, lo, hi = bucket_bootstrap(gap_from_q1, *band_sums(frames[m, t].dropna(subset=[col]), col), rng)
                ax.errorbar(np.arange(4) + (k - 1) * 0.09, point, yerr=[lo, hi], color=color, marker=marker,
                            ls=ls, lw=1.1, ms=3.5, capsize=1.5, elinewidth=0.7, label=name)
            ax.set_title(title)
            ax.set_xticks(range(4), QUARTILES)
        for ax in axes[1]:
            ax.set_xlabel("CH quartile")
        fig.supylabel("Change from Q1 (percentage points)", fontsize=7.5)
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper center", ncol=3, frameon=False)
        fig.savefig(FIG, metadata={"CreationDate": None})
        plt.close(fig)
    print(f"-> {FIG}")


def bucket_figure(d: pd.DataFrame) -> None:
    """Every solution's CH, one column per bucket ordered by rating, with the rating and CH quartile cuts."""
    b = d.groupby("bucket").agg(rating=("rating", "first"), lang=("lang", "first")).sort_values(["rating", "lang"])
    pos = {k: i for i, k in enumerate(b.index)}
    runs = pd.Series(range(len(b)), index=b.index).groupby(rating_quartiles(b["rating"])).agg(["min", "max"])
    rng = np.random.default_rng(20260927)
    with mpl.rc_context(RC):
        fig, (ax, ar) = plt.subplots(2, 1, figsize=(372 / 72, 3.0), sharex=True, layout="constrained",
                                     gridspec_kw={"height_ratios": [3, 1]})
        halo = [pe.withStroke(linewidth=2.6, foreground="white")]
        ax.axhline(8.0, color="black", lw=1.2, ls=(0, (5, 2)), zorder=3, path_effects=halo)
        edges = CH_QUARTILE_EDGES[1:-1]
        for y in edges:
            if y != 8.0:
                ax.axhline(y, color="black", lw=1.2, ls=(0, (1.5, 1.5)), zorder=3, path_effects=halo)
        for i, (lo, hi) in enumerate(zip([1.0, *edges], [*edges, 10.2])):
            ax.text(1.01, (lo + hi) / 2, QUARTILES[i], transform=ax.get_yaxis_transform(), ha="left",
                    va="center", fontsize=6.5, color="0.1", clip_on=False)
        for lang, (name, color) in LANG_STYLE.items():
            g = d[d["lang"] == lang]
            x = g["bucket"].map(pos) + rng.uniform(-0.28, 0.28, len(g))
            ax.scatter(x, g["ch"], s=1.5, color=color, alpha=0.45, lw=0, zorder=2, label=name)
            bb = b[b["lang"] == lang]
            ar.bar([pos[k] for k in bb.index], bb["rating"], width=0.8, color=color, alpha=0.8)
        ax.set_ylabel("CH")
        ax.set_ylim(1, 10.2)
        ax.grid(True, axis="y", color="0.92", lw=0.5)
        ax.set_axisbelow(True)
        for q, (lo, hi) in runs.iterrows():
            if q:
                for a in (ax, ar):
                    a.axvline(lo - 0.5, color="0.2", lw=0.9, zorder=3)
            ar.text((lo + hi) / 2, 3350, f"Q{q + 1}", ha="center", va="top", fontsize=7, color="0.1")
        ar.set_ylabel("Rating")
        ar.set_ylim(0, 3400)
        ar.set_xticks([])
        ar.set_xlim(-0.8, len(b) - 0.2)
        ar.set_xlabel("Buckets, ordered by difficulty rating")
        fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=2, frameon=False, markerscale=5)
        fig.savefig(BUCKET_FIG, metadata={"CreationDate": None})
        plt.close(fig)
    print(f"-> {BUCKET_FIG}")


def rho_figure(frames: dict) -> None:
    """Every bucket's ρ(CH, outcome): the spread behind tab:tasks' Within and Pos."""
    rng = np.random.default_rng(20260926)
    with mpl.rc_context(RC):
        fig, ax = plt.subplots(figsize=(372 / 72, 2.5), layout="constrained")
        ax.axhline(0, color="0.4", lw=0.7, zorder=1)
        ax.grid(True, axis="y", color="0.92", lw=0.5)
        ax.set_axisbelow(True)
        for i, (t, c, _) in enumerate(RHO_OUTCOMES):
            if i:
                ax.axvline(i - 0.5, color="0.85", lw=0.6)
            for k, (m, name) in enumerate(MODELS.items()):
                color, marker, _ = STYLE[m]
                r = np.array(bucket_rhos(frames[m, t], "ch", c))
                x = i + (k - 1) * 0.27
                ax.scatter(x + rng.uniform(-0.08, 0.08, len(r)), r, s=4, color=color, marker=marker,
                           alpha=0.55, lw=0, zorder=2, label=name if i == 0 else None)
                ax.plot([x - 0.11, x + 0.11], [r.mean()] * 2, color="black", lw=1.3, zorder=3,
                        label="mean over buckets" if i == 0 and k == 0 else None)
                ax.text(x, 0.66, f"{100 * (r > 0).mean():.0f}%", ha="center", va="bottom", fontsize=5.5,
                        color=color)
        ax.set_xticks(range(len(RHO_OUTCOMES)), [lab for *_, lab in RHO_OUTCOMES])
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.55, len(RHO_OUTCOMES) - 0.45)
        ax.set_ylim(-0.5, 0.72)
        ax.set_ylabel("Within-bucket $\\rho$(CH, outcome)")
        h, lab = ax.get_legend_handles_labels()
        order = sorted(range(len(lab)), key=lambda j: lab[j] == "mean over buckets")
        fig.legend([h[j] for j in order], [lab[j] for j in order], loc="outside upper center", ncol=4,
                   frameon=False, markerscale=2)
        fig.savefig(RHO_FIG, metadata={"CreationDate": None})
        plt.close(fig)
    print(f"-> {RHO_FIG}")


def main() -> None:
    frames = {(m, t): load_task(m, t) for m in MODELS for t in OUTCOMES}
    corpus = load_corpus()
    print("Corpus and RQ2: task outcomes across CodeHealth")
    corpus_numbers(corpus)
    methods_numbers(frames, corpus)
    tasks_tables(frames)
    bands_table(frames)
    replication_table(frames)
    operator_table(frames)
    example_pair(frames)
    text_numbers(frames)
    print()
    figure(frames)
    rho_figure(frames)
    bucket_figure(corpus)


if __name__ == "__main__":
    main()
