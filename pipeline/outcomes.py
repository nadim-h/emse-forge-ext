"""Load the corpus and each model's task outcomes as tidy frames, one row per
solution (Tasks A, C) or seeded fault (Task B). CH is the CodeScene CLI v1.0.33
score from `data/codehealth/reviews.jsonl`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline import data
from pipeline.data import cli_ch, task_rows
from pipeline.stats import with_bucket

# Model slug -> display name, in the paper's order.
MODELS = {"gemma-4-26b": "Gemma", "qwen3.6-35b": "Qwen", "nemotron-3-nano-30b-a3b": "Nemotron"}
LANG_NAMES = {"cpp": "C++", "java": "Java"}
# Figure style: each model's color, marker and line style, and the matplotlib rc.
STYLE = dict(zip(MODELS, [("#E69F00", "o", "-"), ("#0072B2", "s", "--"), ("#009E73", "^", ":")]))
RC = {"font.size": 7.5, "axes.titlesize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
      "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}


def _solution(lang: str, r: dict) -> dict:
    return {"lang": lang, "problem": r["name"], "solution_id": r["solution_id"], "sloc": r["sloc"]}


def _frame(rows: list[dict]) -> pd.DataFrame:
    d = with_bucket(pd.DataFrame(rows))
    return d.assign(ch=cli_ch(d))


def with_rating(d: pd.DataFrame) -> pd.DataFrame:
    return d.assign(rating=d["problem"].map({k: v["rating"] for k, v in data.ratings().items()}))


def load_testgen(model: str) -> pd.DataFrame:
    """Task A; mutation score and coverage need a valid test."""
    rows = []
    for lang, r in task_rows(model, "testgen"):
        if r.get("ok"):
            cov = r["coverage_valid"]
            rows.append({
                **_solution(lang, r), "validity_rate": r["validity_rate"], "mutation_score": r["mutation_score"],
                "line_rate": cov["line_rate"] if cov["ok"] else np.nan,
                "branch_rate": cov["branch_rate"] if cov["ok"] else np.nan,
                "n_valid": r["n_valid"], "n_mutants": r["n_mutants"], "n_mutants_compiled": r["n_mutants_compiled"],
                "kills": r["kills_by_operator"],
            })
    return _frame(rows)


def load_bugfix(model: str) -> pd.DataFrame:
    return _frame([{**_solution(lang, r), "operator": r["operator"], "fixed": float(r["fixed"])}
                   for lang, r in task_rows(model, "bugfix")])


def load_refactor(model: str) -> pd.DataFrame:
    """Task C; `effective` is preserved behavior with a CH gain."""
    d = _frame([{**_solution(lang, r), "preserved": float(not r["broken"]), "ch_after": r["ch_after"]}
                for lang, r in task_rows(model, "refactor") if r["ok"]])
    d = d.assign(ch_delta=(d["ch_after"] - d["ch"]).round(2))
    return d.assign(effective=((d["preserved"] == 1) & (d["ch_delta"] > 0)).astype(float))


def load_task(model: str, task: str) -> pd.DataFrame:
    return {"A": load_testgen, "B": load_bugfix, "C": load_refactor}[task](model)


def load_corpus() -> pd.DataFrame:
    """Every solution with its number of tests and its problem's rating."""
    return with_rating(_frame([{**_solution(lang, r), "n_oracle": r["n_tests"]}
                               for lang in data.LANGS for r in data.load_dataset(lang)]))


def print_table(title: str, columns: list[str], rows: list[list]) -> None:
    print(f"\n{title}:")
    print(pd.DataFrame(rows, columns=columns).to_string(index=False))
