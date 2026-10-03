"""Task C: have the model refactor each solution, then judge behavior
preservation on the oracle and score the output's CH with the `cs` CLI.

Usage:
    python -m pipeline.refactor generate
    python -m pipeline.refactor evaluate
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil

from pipeline import config as C
from pipeline import data
from pipeline import exec_harness as EX
from pipeline import llm
from pipeline import oracle as O
from pipeline import prompts
from pipeline import runner
from pipeline import study as S

# The CodeScene CLI, v1.0.33.
CS_BIN = (os.environ.get("EMSE_CS_BIN") or shutil.which("cs")
          or os.path.expanduser("~/.local/bin/cs"))
CS_ENV = {"CS_DISABLE_VERSION_CHECK": "1"}
REVIEW_TIMEOUT = 40.0


def code_health(source: str, lang: str) -> float | None:
    """The `cs review` score, or None on a timeout or bad output."""
    r = EX.run_process(
        [CS_BIN, "review", "--output-format", "json", "--file-name", f"s.{lang}"],
        stdin_data=source, timeout=REVIEW_TIMEOUT, env=CS_ENV)
    try:
        return json.loads(r.stdout).get("score")
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------- generate ---
async def generate(concurrency: int, resume: bool) -> None:
    cli, prompt = llm.client(), prompts.load("refactor")
    for lang in C.LANGS:
        async def one(row):
            msgs = prompt.messages(langname=C.LANGNAME[row["lang"]], lang=row["lang"],
                                   code=row["task"])
            # A reply without a code block is also resampled.
            code, attempts = await llm.complete_parsed(
                cli, msgs, lambda t: llm.extract_code(t, row["lang"]), retry_on_parse_miss=True)
            return {"solution_id": row["solution_id"], "lang": row["lang"],
                    "name": row["name"], "score": row["score"], "sloc": row["sloc"],
                    "has_code": bool(code), "refactored_source": code or "",
                    "attempts": attempts}

        out = await runner.run_llm(
            data.load_dataset(lang), one, key_of=lambda r: r["solution_id"],
            out_path=S.path("refactor", f"{lang}_outputs.jsonl"),
            concurrency=concurrency, resume=resume, keep=lambda r: r.get("has_code", False))
        print(f"[{lang}] {sum(r.get('has_code', False) for r in out)}/{len(out)} produced code")


# --------------------------------------------------------------- evaluate ---
def evaluate_one(row: dict, src: str, judge: O.Oracle, prev_ch: float | None) -> dict:
    result = {
        "solution_id": row["solution_id"], "lang": row["lang"], "name": row["name"],
        "score": row["score"], "sloc": row["sloc"],
        "unchanged": bool(src) and src.strip() == row["task"].strip(),
        "compiled": False, "broken": True, "n_tests": 0, "n_pass": 0,
        "ch_after": None, "ok": True, "oracle": O.ORACLE_TAG,
    }
    if not src:
        result["error"] = "no code returned"
        return result
    result.update(judge.judge_source(row["lang"], src, row["name"]))
    if result["compiled"]:
        result["broken"] = result["n_pass"] < result["n_tests"]
        # A re-judge keeps the CH on file rather than re-running `cs`.
        result["ch_after"] = prev_ch if prev_ch is not None else code_health(src, row["lang"])
    return result


def evaluate(workers: int, resume: bool) -> None:
    """Judge every stored refactor."""
    for lang in C.LANGS:
        judge = O.Oracle(lang)
        # A reply without code is scored as broken.
        srcs = {o["solution_id"]: o.get("refactored_source", "")
                for o in runner.read_jsonl(S.path("refactor", f"{lang}_outputs.jsonl"))}
        out_path = S.path("refactor", f"{lang}_results.jsonl")
        prev = {r["solution_id"]: r.get("ch_after") for r in runner.read_jsonl(out_path)}
        out = runner.run_parallel(
            [(r, srcs[r["solution_id"]], judge, prev.get(r["solution_id"]))
             for r in data.load_dataset(lang) if r["solution_id"] in srcs],
            lambda j: evaluate_one(*j), key_of=lambda j: j[0]["solution_id"], out_path=out_path,
            workers=workers, resume=resume,
            keep=lambda r: r.get("ok", False) and r.get("oracle") == O.ORACLE_TAG)
        ok = [r for r in out if r.get("ok")]
        broken = sum(r["broken"] for r in ok)
        print(f"[{lang}] {len(ok)}/{len(out)} evaluated, {broken} broken")


def main() -> None:
    ap = argparse.ArgumentParser(description="Task C for the active model (EMSE_MODEL_PROFILE).")
    ap.add_argument("cmd", choices=["generate", "evaluate"])
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--workers", type=int, default=22)
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()
    if a.cmd == "generate":
        asyncio.run(generate(a.concurrency, not a.no_resume))
    else:
        evaluate(a.workers, not a.no_resume)


if __name__ == "__main__":
    main()
