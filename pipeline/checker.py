"""Each problem's testlib checker: compile it and judge an output with it."""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pipeline import config as C
from pipeline import exec_harness as EX

CHECKER_TIMEOUT = 30.0
TESTLIB_DIR = C.ROOT / "third_party" / "testlib"
CHECKER_DIR = C.WORK_DIR / "checkers"


def _compile_checker(source: str) -> Path | None:
    """One problem's testlib checker, compiled once and cached by content; None if
    it does not compile."""
    testlib = (TESTLIB_DIR / "testlib.h").read_text(encoding="utf-8")
    d = CHECKER_DIR / hashlib.sha1((source + testlib).encode("utf-8")).hexdigest()[:16]
    exe = d / "check"
    if exe.exists():
        return exe
    d.mkdir(parents=True, exist_ok=True)
    (d / "check.cpp").write_text(source, encoding="utf-8")
    # Built under a per-process name, so concurrent runs never see half a binary.
    tmp = d / f"check.{os.getpid()}.tmp"
    res = EX.run_process(["g++", "-std=gnu++17", "-O2", "-w", f"-I{TESTLIB_DIR}",
                          str(d / "check.cpp"), "-o", str(tmp)],
                         timeout=C.COMPILE_TIMEOUT * 3, cwd=d)
    if res.status != EX.OK:
        return None
    os.replace(tmp, exe)
    return exe


def compile_checkers(problems: list[dict]) -> dict[str, Path | None]:
    """name -> checker binary, None where it does not compile."""
    sources = sorted({p["checker"] for p in problems})
    with ThreadPoolExecutor(max_workers=8) as ex:
        exes = dict(zip(sources, ex.map(_compile_checker, sources)))
    out = {p["name"]: exes[p["checker"]] for p in problems}
    if bad := sorted(n for n, e in out.items() if e is None):
        print(f"  judged without a checker: {bad}", file=sys.stderr)
    return out


def accepts(checker: Path | None, test: dict, stdout: str) -> bool:
    """Whether `stdout` is accepted on `test`; an exact match skips the checker, and a
    checker that times out or fails counts as a rejection."""
    if EX.normalize(stdout) == EX.normalize(test["output"]):
        return True
    if checker is None:
        return EX.outputs_match(stdout, test["output"])
    d = Path(tempfile.mkdtemp(prefix="chk_", dir=C.WORK_DIR))
    try:
        for fn, text in (("in", test["input"]), ("out", stdout), ("ans", test["output"])):
            (d / fn).write_text(text, encoding="utf-8")
        r = EX.run_process([str(checker), str(d / "in"), str(d / "out"), str(d / "ans")],
                           timeout=CHECKER_TIMEOUT, cwd=d)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    # testlib exits 0 only on an accepted output.
    return r.status != EX.TIMEOUT and r.returncode == 0
