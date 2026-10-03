"""The statistics every report shares.

A bucket is one problem in one language (`with_bucket`); Codeforces Div. 1/Div. 2
mirrors fold onto one canonical problem within a language.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

from pipeline import data

# The corpus CH quartiles (7.47, 7.995, 8.54), each rounded up to one decimal.
CH_QUARTILE_EDGES = [0, 7.5, 8.0, 8.6, 10.01]
QUARTILES = ["Q1", "Q2", "Q3", "Q4"]
HEALTHY = 9.0  # the conference paper's Healthy line
N_BOOT = 2000  # bootstrap resamples


def with_bucket(df: pd.DataFrame) -> pd.DataFrame:
    canon = {name: r["canonical"] for name, r in data.ratings().items()}
    return df.assign(bucket=df["lang"] + "::" + df["problem"].map(canon))


def ch_bands(ch: pd.Series) -> pd.Series:
    return pd.cut(ch, CH_QUARTILE_EDGES, labels=QUARTILES, right=False)


def rating_quartiles(rating: pd.Series) -> pd.Series:
    """Quartile (0 = easiest) of one rating per bucket; tied ratings share a quartile."""
    return pd.qcut(rating, 4, labels=False)


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def holm(pvalues) -> np.ndarray:
    """Holm-adjusted p-values in input order."""
    return multipletests(pvalues, method="holm")[1]


def pooled_spearman(df: pd.DataFrame, x: str, y: str) -> float:
    d = df[[x, y]].dropna()
    return stats.spearmanr(d[x], d[y])[0]


def summarize(rhos: list[float]) -> dict:
    rhos = np.array(rhos)
    n_pos = int((rhos > 0).sum())
    return dict(mean_rho=float(rhos.mean()), k=len(rhos), pos=n_pos / len(rhos),
                sign_p=float(stats.binomtest(n_pos, len(rhos), 0.5).pvalue))


def bucket_rhos(df: pd.DataFrame, x: str, y: str) -> list[float]:
    """Spearman of y on x inside each bucket where y varies."""
    d = df[[x, y, "bucket"]].dropna()
    return [stats.spearmanr(g[x], g[y]).correlation for _, g in d.groupby("bucket") if g[y].nunique() > 1]


def within(df: pd.DataFrame, x: str, y: str) -> dict:
    """Mean within-bucket Spearman, with a sign test over buckets."""
    return summarize(bucket_rhos(df, x, y))


def bucket_partial_rhos(df: pd.DataFrame, x: str, y: str, controls: list[str]) -> list[float]:
    """Per-bucket Spearman partial correlation of x and y, the controls' ranks regressed out."""
    out = []
    for _, g in df.groupby("bucket"):
        r = np.column_stack([stats.rankdata(g[v]) for v in (x, y, *controls)])
        z = np.column_stack([np.ones(len(g)), r[:, 2:]])
        res = r[:, :2] - z @ np.linalg.lstsq(z, r[:, :2], rcond=None)[0]
        out.append(float(np.corrcoef(res.T)[0, 1]))
    return out


def band_sums(d: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    """Sum and count of `col` per bucket (row) and CH quartile (column)."""
    agg = (d.assign(band=ch_bands(d["ch"]).cat.codes).groupby(["bucket", "band"])[col]
           .agg(["sum", "count"]).unstack(fill_value=0))
    return agg["sum"].to_numpy(float), agg["count"].to_numpy(float)


def bucket_bootstrap(stat, s: np.ndarray, c: np.ndarray, rng):
    """`stat(s, c)` and the distances to its 95% interval from resampling buckets (rows)."""
    point = stat(s, c)
    boot = np.array([stat(s[i], c[i]) for i in rng.integers(0, len(s), (N_BOOT, len(s)))])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return point, point - lo, hi - point


def fit_clustered(d: pd.DataFrame, formula: str):
    """OLS with standard errors clustered by bucket; `d` must have no missing values."""
    return smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["bucket"]})


def fixed_effects(df: pd.DataFrame, outcome: str) -> dict:
    """Change in `outcome` per CH point, with bucket intercepts and log(SLOC); a linear
    probability model for binary outcomes."""
    d = df[["ch", "sloc", outcome, "bucket"]].dropna()
    d = d.assign(log_sloc=np.log(d["sloc"]), y=d[outcome])
    m = fit_clustered(d, "y ~ ch + log_sloc + C(bucket)")
    return {"coef": m.params["ch"], "p": m.pvalues["ch"]}
