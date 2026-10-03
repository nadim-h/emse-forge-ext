"""Paths and loaders for the committed data."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGURES_DIR = ROOT / "figures"
LANGS = ("cpp", "java")
# Upper-tail PPL outlier cut (one-sided robust z), as in the conference paper.
ROBUST_Z = 2.5


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def load_dataset(lang: str) -> list[dict]:
    """The 6,198 verified solutions."""
    return read_jsonl(DATA / "dataset" / f"{lang}_dataset.jsonl")


def load_descriptions(lang: str) -> dict[str, str]:
    """Problem name -> statement."""
    return {r["name"]: r["description"]
            for r in read_jsonl(DATA / "problems" / f"{lang}_descriptions.jsonl")}


def load_tests(lang: str) -> dict[str, dict]:
    """Problem name -> {"checker", "tests", ...}: a seeded sample of up to 100
    CodeContests+ (`ccplus_5x`) tests, less 105 that no standard judgment can
    handle; `ccplus_indices` is each test's position in CodeContests+."""
    return {r["name"]: r for r in read_jsonl(DATA / "problems" / f"{lang}_tests.jsonl")}


def task_rows(model: str, task: str):
    """(lang, row) of one model's judged results for one task directory."""
    for lang in LANGS:
        for r in read_jsonl(DATA / f"{task}_{model}" / f"{lang}_results.jsonl"):
            yield lang, r


def ratings() -> dict[str, dict]:
    """Problem name -> rating, rating source and canonical (mirror-folded) id."""
    return {r["name"]: r for r in read_jsonl(DATA / "problems" / "problem_difficulty.jsonl")}


def cli_ch(d: pd.DataFrame) -> pd.Series:
    """CH from the CodeScene CLI v1.0.33 review of each solution."""
    scores = {r["solution_id"]: r["cs_score"] for r in read_jsonl(DATA / "codehealth" / "reviews.jsonl")}
    return d["solution_id"].map(scores)


def load_ppl(model: str) -> pd.DataFrame:
    """One model's PPL for every dataset solution, with the CLI's CH."""
    d = pd.DataFrame(read_jsonl(DATA / "perplexity" / f"{model}.jsonl"))
    d = d[np.isfinite(d["ppl"])]
    return d.assign(ch=cli_ch(d))


def filter_outliers(d: pd.DataFrame) -> pd.DataFrame:
    """Drop the upper tail by one-sided robust (MAD) z-score, within each language."""
    def z(x):
        dev = x - x.median()
        return 0.6745 * dev / dev.abs().median()
    return d[d.groupby("lang")["ppl"].transform(z) <= ROBUST_Z]
