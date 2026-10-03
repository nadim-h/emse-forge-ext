"""Line and branch coverage of a test suite: gcov for C++, JaCoCo for Java.
Each program is built once, run on every input, and the totals read at the end."""
from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass

from pipeline import config as C
from pipeline import exec_harness as EX

JACOCO_AGENT = C.ROOT / "tools" / "jacocoagent-real.jar"
JACOCO_CLI = C.ROOT / "tools" / "jacococli.jar"


@dataclass
class Coverage:
    lines_total: int = 0
    lines_covered: int = 0
    branches_total: int = 0
    branches_covered: int = 0
    ok: bool = False
    error: str = ""

    @property
    def line_rate(self) -> float:
        return self.lines_covered / self.lines_total if self.lines_total else 0.0

    @property
    def branch_rate(self) -> float:
        return self.branches_covered / self.branches_total if self.branches_total else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["line_rate"] = round(self.line_rate, 4)
        d["branch_rate"] = round(self.branch_rate, 4)
        return d


def measure(lang: str, source: str, inputs: list[str]) -> Coverage:
    """Coverage of `source` over `inputs`; errors are reported in `Coverage.error`."""
    return _measure_cpp(source, inputs) if lang == "cpp" else _measure_java(source, inputs)


# --------------------------------------------------------------------- C++ --
_GCOV_LINES = re.compile(r"Lines executed:([\d.]+)% of (\d+)")
_GCOV_BRANCH = re.compile(r"Branches executed:([\d.]+)% of (\d+)")
_GCOV_TAKEN = re.compile(r"Taken at least once:([\d.]+)% of (\d+)")


def _measure_cpp(source: str, inputs: list[str]) -> Coverage:
    prog, err = EX.compile_program("cpp", source, coverage=True)
    if prog is None:
        return Coverage(error=f"compile: {err[:200]}")
    try:
        for data in inputs:
            EX.run_program(prog, data)
        g = EX.run_process(["gcov", "-b", "-n", "main.cpp"],
                           cwd=prog.workdir, timeout=60)
        text = g.stdout + g.stderr
        cov = Coverage()
        if m := _GCOV_LINES.search(text):
            cov.lines_total = int(m.group(2))
            cov.lines_covered = round(float(m.group(1)) / 100 * cov.lines_total)
        if m := _GCOV_BRANCH.search(text):
            cov.branches_total = int(m.group(2))
        if m := _GCOV_TAKEN.search(text):
            # "Taken at least once" is branch coverage; "Branches executed" is not.
            cov.branches_covered = round(float(m.group(1)) / 100 * cov.branches_total)
        cov.ok = cov.lines_total > 0
        if not cov.ok:
            cov.error = f"gcov produced no totals: {text[:200]}"
        return cov
    finally:
        prog.cleanup()


# -------------------------------------------------------------------- Java --
def _measure_java(source: str, inputs: list[str]) -> Coverage:
    prog, err = EX.compile_program("java", source)
    if prog is None:
        return Coverage(error=f"compile: {err[:200]}")
    try:
        exec_file = prog.workdir / "jacoco.exec"
        agent = (f"-javaagent:{JACOCO_AGENT}=destfile={exec_file},"
                 f"append=true,output=file,dumponexit=true")
        argv = [prog.argv[0], agent, *prog.argv[1:]]
        for data in inputs:
            EX.run_process(argv, stdin_data=data, cwd=prog.workdir)
        # Runs killed at the time limit write no coverage.
        if not exec_file.exists():
            return Coverage(error="jacoco produced no exec file")

        csv_out = prog.workdir / "cov.csv"
        r = EX.run_process([
            "java", "-jar", str(JACOCO_CLI), "report", str(exec_file),
            "--classfiles", str(prog.workdir), "--csv", str(csv_out),
        ], timeout=120, cwd=prog.workdir)
        if not csv_out.exists():
            return Coverage(error=f"jacococli: {(r.stderr or r.stdout)[:200]}")
        cov = Coverage()
        with csv_out.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                lm, lc = int(row["LINE_MISSED"]), int(row["LINE_COVERED"])
                bm, bc = int(row["BRANCH_MISSED"]), int(row["BRANCH_COVERED"])
                cov.lines_total += lm + lc
                cov.lines_covered += lc
                cov.branches_total += bm + bc
                cov.branches_covered += bc
        cov.ok = cov.lines_total > 0
        if not cov.ok:
            cov.error = "empty jacoco report"
        return cov
    finally:
        prog.cleanup()
