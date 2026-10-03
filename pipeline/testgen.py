"""Task A: the model writes up to n tests from the code alone; the suites are
scored by validity, mutation score and coverage.

Usage:
    python -m pipeline.testgen generate
    python -m pipeline.testgen evaluate
"""
from __future__ import annotations

import argparse
import asyncio
import statistics as stat
from collections import Counter

from pipeline import config as C
from pipeline import coverage as COV
from pipeline import data
from pipeline import exec_harness as EX
from pipeline import llm
from pipeline import mutation
from pipeline import prompts
from pipeline import runner
from pipeline import study as S


def _parse_tests(text: str) -> list[dict]:
    parsed = llm.extract_json(text)
    if not isinstance(parsed, list):
        return []
    out = []
    for t in parsed:
        if isinstance(t, dict) and "input" in t and "output" in t:
            inp, exp = str(t["input"]), str(t["output"])
            # The prompt's skeleton echoed back is not a test.
            if inp == "..." and exp == "...":
                continue
            out.append({"input": inp, "output": exp})
    return out


# --------------------------------------------------------------- generate ---
async def generate(concurrency: int, resume: bool) -> None:
    cli, prompt = llm.client(), prompts.load("testgen")
    for lang in C.LANGS:
        async def one(row):
            msgs = prompt.messages(lang=row["lang"], langname=C.LANGNAME[row["lang"]],
                                   code=row["task"], n=S.GENERATED_TESTS_PER_SOLUTION)
            tests, _ = await llm.complete_parsed(cli, msgs, _parse_tests)
            return {"solution_id": row["solution_id"], "lang": row["lang"],
                    "name": row["name"], "score": row["score"], "sloc": row["sloc"],
                    "parse_ok": bool(tests), "tests": tests or []}

        out = await runner.run_llm(
            data.load_dataset(lang), one, key_of=lambda r: r["solution_id"],
            out_path=S.path("testgen", f"{lang}_suites.jsonl"),
            concurrency=concurrency, resume=resume, keep=lambda r: r.get("parse_ok", False))
        print(f"[{lang}] {sum(r.get('parse_ok', False) for r in out)}/{len(out)} suites")


# --------------------------------------------------------------- evaluate ---
def _valid_tests(prog: EX.Program, tests: list[dict]) -> tuple[list[dict], float]:
    """The tests `prog` passes, with its own output as the expected one, and its
    slowest run on them."""
    valid, slowest = [], 0.0
    for t in tests:
        r = EX.run_program(prog, t["input"])
        if r.status == EX.OK and EX.outputs_match(r.stdout, t["output"]):
            slowest = max(slowest, r.elapsed)
            valid.append({"input": t["input"], "output": r.stdout})
    return valid, slowest


def _mutation(lang: str, row: dict, valid: list[dict], slowest: float) -> dict:
    """How many of the solution's compiling mutants the valid tests kill, per operator."""
    muts = mutation.generate_mutants(row["task"], lang, limit=S.MUTANTS_PER_SOLUTION)
    timeout = mutation.mutant_timeout(slowest)
    total, killed = Counter(), Counter()
    for m in muts:
        mp, _ = EX.compile_program(lang, m.source, std_hint=row["std"])
        if mp is None:
            continue
        total[m.operator] += 1
        try:
            killed[m.operator] += mutation.kills(mp, valid, timeout)
        finally:
            mp.cleanup()
    n_compiled, n_killed = total.total(), killed.total()
    return {"n_mutants": len(muts), "n_mutants_compiled": n_compiled, "n_killed_valid": n_killed,
            "mutation_score": round(n_killed / n_compiled, 4) if n_compiled else None,
            "kills_by_operator": {k: [killed[k], total[k]] for k in total}}


def evaluate_one(row: dict, tests: list[dict]) -> dict:
    lang = row["lang"]
    out = {"solution_id": row["solution_id"], "lang": lang, "name": row["name"],
           "score": row["score"], "sloc": row["sloc"], "n_generated": len(tests), "ok": False}
    prog, err = EX.compile_program(lang, row["task"], std_hint=row["std"])
    if prog is None:
        out["error"] = f"compile: {err[:200]}"
        return out
    try:
        valid, slowest = _valid_tests(prog, tests)
    finally:
        prog.cleanup()
    out.update(n_valid=len(valid), validity_rate=round(len(valid) / len(tests), 4), ok=True)
    if not valid:
        out.update(n_mutants=0, n_mutants_compiled=0, n_killed_valid=0,
                   mutation_score=None, kills_by_operator={},
                   coverage_valid=COV.Coverage(error="no valid tests").to_dict())
        return out
    out.update(_mutation(lang, row, valid, slowest))
    out["coverage_valid"] = COV.measure(lang, row["task"], [t["input"] for t in valid]).to_dict()
    return out


def evaluate(workers: int, resume: bool) -> None:
    for lang in C.LANGS:
        suites = {g["solution_id"]: g["tests"][:S.GENERATED_TESTS_PER_SOLUTION]
                  for g in runner.read_jsonl(S.path("testgen", f"{lang}_suites.jsonl"))
                  if g.get("parse_ok")}
        jobs = [(r, suites[r["solution_id"]])
                for r in data.load_dataset(lang) if r["solution_id"] in suites]
        out = runner.run_parallel(
            jobs, lambda j: evaluate_one(*j), key_of=lambda j: j[0]["solution_id"],
            out_path=S.path("testgen", f"{lang}_results.jsonl"),
            workers=workers, resume=resume, keep=lambda r: r.get("ok", False))
        scores = [r["mutation_score"] for r in out
                  if r.get("ok") and r["mutation_score"] is not None]
        print(f"[{lang}] {sum(r.get('ok', False) for r in out)}/{len(out)} evaluated, "
              f"mean mutation score {stat.mean(scores):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Task A for the active model (EMSE_MODEL_PROFILE).")
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
