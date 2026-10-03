"""Task B: seed faults from Task A's mutants, have the model repair them, judge the repairs.

A mutant becomes a fault if it (i) compiles and (ii) fails one of its problem's tests, up
to two per solution from (iii) distinct operator families. A repair succeeds if the
oracle accepts it on every test.

Usage:
    python -m pipeline.bugfix seed
    python -m pipeline.bugfix repair
    python -m pipeline.bugfix verify
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from pipeline import config as C
from pipeline import data
from pipeline import exec_harness as EX
from pipeline import llm
from pipeline import mutation
from pipeline import oracle as O
from pipeline import prompts
from pipeline import runner
from pipeline import study as S


def faults_path(lang: str) -> Path:
    """The seeded faults, shared by every model."""
    return C.DATA / "bugfix" / f"{lang}_faults.jsonl"


# ------------------------------------------------------------------- seed ---
def _reference(prog: EX.Program, tests: list[dict]) -> tuple[list[dict], float]:
    """The solution's own output on each test it runs without error, and its slowest run."""
    ref, slowest = [], 0.0
    for t in tests:
        r = EX.run_program(prog, t["input"])
        if r.status == EX.OK:
            slowest = max(slowest, r.elapsed)
            ref.append({"input": t["input"], "output": r.stdout})
    return ref, slowest


def seed_one(row: dict, tests: list[dict]) -> dict:
    lang, std = row["lang"], row["std"]
    out = {"solution_id": row["solution_id"], "name": row["name"],
           "score": row["score"], "sloc": row["sloc"], "faults": []}
    prog, err = EX.compile_program(lang, row["task"], std_hint=std)
    if prog is None:
        out["error"] = f"compile: {err[:200]}"
        return out
    try:
        ref, slowest = _reference(prog, tests)
    finally:
        prog.cleanup()

    timeout, seen_ops = mutation.mutant_timeout(slowest), set()
    for m in mutation.generate_mutants(row["task"], lang, limit=S.MUTANTS_PER_SOLUTION):
        if len(out["faults"]) >= S.FAULTS_PER_SOLUTION:
            break
        if m.operator in seen_ops:
            continue
        mp, _ = EX.compile_program(lang, m.source, std_hint=std)
        if mp is None:
            continue
        try:
            observable = mutation.kills(mp, ref, timeout)
        finally:
            mp.cleanup()
        if observable:
            seen_ops.add(m.operator)
            out["faults"].append({"fault_id": f"{row['solution_id']}::{m.mutant_id}",
                                  "operator": m.operator, "line": m.line,
                                  "buggy_source": m.source})
    return out


def seed(workers: int, resume: bool) -> None:
    for lang in C.LANGS:
        tests = data.load_tests(lang)
        jobs = [(r, tests[r["name"]]["tests"]) for r in data.load_dataset(lang)]
        out = runner.run_parallel(
            jobs, lambda j: seed_one(*j), key_of=lambda j: j[0]["solution_id"],
            out_path=faults_path(lang), workers=workers, resume=resume,
            keep=lambda r: bool(r.get("faults")))
        out = [r for r in out if r.get("faults")]
        runner.write_jsonl(faults_path(lang), out)
        print(f"[{lang}] {sum(len(r['faults']) for r in out)} faults across {len(out)} solutions")


# ----------------------------------------------------------------- repair ---
def _repair_jobs(lang: str) -> list[dict]:
    descs = data.load_descriptions(lang)
    return [{**f, "solution_id": sol["solution_id"], "lang": lang, "name": sol["name"],
             "score": sol["score"], "sloc": sol["sloc"], "description": descs[sol["name"]]}
            for sol in data.read_jsonl(faults_path(lang)) for f in sol["faults"]]


async def repair(concurrency: int, resume: bool) -> None:
    cli, prompt = llm.client(), prompts.load("bugfix")
    for lang in C.LANGS:
        async def one(job):
            msgs = prompt.messages(lang=job["lang"], langname=C.LANGNAME[job["lang"]],
                                   description=job["description"], code=job["buggy_source"])
            code, _ = await llm.complete_parsed(
                cli, msgs, lambda t: llm.extract_code(t, job["lang"]))
            return {
                "fault_id": job["fault_id"], "solution_id": job["solution_id"],
                "lang": job["lang"], "name": job["name"], "score": job["score"],
                "sloc": job["sloc"], "operator": job["operator"],
                "line": job["line"], "has_code": bool(code),
                "unchanged": bool(code) and code.strip() == job["buggy_source"].strip(),
                "fixed_source": code or "",
            }

        out = await runner.run_llm(
            _repair_jobs(lang), one, key_of=lambda j: j["fault_id"],
            out_path=S.path("bugfix", f"{lang}_repairs.jsonl"),
            concurrency=concurrency, resume=resume, keep=lambda r: r.get("has_code", False))
        print(f"[{lang}] {sum(r.get('has_code', False) for r in out)}/{len(out)} produced code")


# ----------------------------------------------------------------- verify ---
def verify_one(rep: dict, row: dict, judge: O.Oracle) -> dict:
    out = {k: rep[k] for k in ("fault_id", "solution_id", "lang", "name", "score", "sloc",
                               "operator", "has_code", "unchanged")}
    out.update(compiled=False, fixed=False, n_pass=0, n_tests=0, oracle=O.ORACLE_TAG)
    if not rep["fixed_source"]:
        out["error"] = "no code returned"
        return out
    out.update(judge.judge_source(rep["lang"], rep["fixed_source"], rep["name"], row["std"]))
    out["fixed"] = out["compiled"] and out["n_pass"] == out["n_tests"]
    return out


def verify(workers: int, resume: bool) -> None:
    """Judge every stored repair."""
    for lang in C.LANGS:
        judge = O.Oracle(lang)
        rows = {r["solution_id"]: r for r in data.load_dataset(lang)}
        reps = runner.read_jsonl(S.path("bugfix", f"{lang}_repairs.jsonl"))
        out = runner.run_parallel(
            [(rep, rows[rep["solution_id"]], judge) for rep in reps],
            lambda j: verify_one(*j), key_of=lambda j: j[0]["fault_id"],
            out_path=S.path("bugfix", f"{lang}_results.jsonl"),
            workers=workers, resume=resume, keep=lambda r: r.get("oracle") == O.ORACLE_TAG)
        print(f"[{lang}] {sum(r.get('fixed', False) for r in out)}/{len(out)} repaired")


def main() -> None:
    ap = argparse.ArgumentParser(description="Task B for the active model (EMSE_MODEL_PROFILE); "
                                             "`seed` is shared by every model.")
    ap.add_argument("cmd", choices=["seed", "repair", "verify"])
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--workers", type=int, default=22)
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()
    if a.cmd == "seed":
        seed(a.workers, not a.no_resume)
    elif a.cmd == "repair":
        asyncio.run(repair(a.concurrency, not a.no_resume))
    else:
        verify(a.workers, not a.no_resume)


if __name__ == "__main__":
    main()
