"""The judge of Tasks B and C: a program passes when its
problem's checker accepts it on every one of the problem's tests (`data.load_tests`).
"""
from __future__ import annotations

from pipeline import checker as K
from pipeline import data
from pipeline import exec_harness as EX

# As on Codeforces; deeply recursive accepted solutions crash at the OS default.
CPP_STACK_MB = 256
# After this many timeouts the remaining tests count as failed, unrun.
MAX_TIMEOUTS = 3
# Stamped on every Task B/C verdict; verdicts with another tag are recomputed.
ORACLE_TAG = "ccplus_5x-100-checker-screened"


class Oracle:
    """One language's problems, tests and compiled checkers."""

    def __init__(self, lang: str):
        self.probs = data.load_tests(lang)
        self.checkers = K.compile_checkers(list(self.probs.values()))

    def judge(self, prog: EX.Program, problem: str) -> dict:
        """{"n_tests", "n_pass", "n_timeout"} for `prog` on `problem`'s tests."""
        tests, checker = self.probs[problem]["tests"], self.checkers[problem]
        n_pass = n_timeout = 0
        for t in tests:
            if n_timeout >= MAX_TIMEOUTS:
                break
            r = EX.run_program(prog, t["input"])
            if r.status == EX.TIMEOUT:
                n_timeout += 1
            elif r.status == EX.OK and K.accepts(checker, t, r.stdout):
                n_pass += 1
        return {"n_tests": len(tests), "n_pass": n_pass, "n_timeout": n_timeout}

    def judge_source(self, lang: str, src: str, problem: str, std_hint: str | None = None) -> dict:
        """Compile `src` and judge it on `problem`'s tests."""
        prog, err = EX.compile_program(lang, src, std_hint=std_hint, stack_mb=CPP_STACK_MB)
        if prog is None:
            return {"compiled": False, "error": f"compile: {err[:200]}"}
        try:
            return {"compiled": True, **self.judge(prog, problem)}
        finally:
            prog.cleanup()
