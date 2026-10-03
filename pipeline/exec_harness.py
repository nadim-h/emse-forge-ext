"""Compile and run single-file C++ and Java programs.

Each subprocess runs in its own session, and on timeout the whole process group
is killed. C++ memory is capped with `ulimit -v`, Java's with -Xmx/-Xss.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from pipeline import config as C

OK = "ok"
TIMEOUT = "timeout"
RUNTIME_ERROR = "runtime_error"
OUTPUT_LIMIT = "output_limit"


@dataclass
class RunResult:
    status: str
    stdout: str
    stderr: str
    returncode: int
    elapsed: float


@dataclass
class Program:
    """A compiled program plus the directory that owns its artifacts."""
    workdir: Path
    argv: list[str]

    def cleanup(self) -> None:
        shutil.rmtree(self.workdir, ignore_errors=True)


# ------------------------------------------------------------- processes ---
def run_process(argv: list[str], *, stdin_data: str = "", timeout: float = C.RUN_TIMEOUT,
                cwd: Path | None = None, env: dict | None = None) -> RunResult:
    """Run argv to completion in its own session; on timeout the whole group is killed."""
    t0 = time.monotonic()
    proc = subprocess.Popen(
        argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=cwd, env={**os.environ, **(env or {})}, start_new_session=True,
        text=True, errors="replace")
    try:
        out, err = proc.communicate(input=stdin_data, timeout=timeout)
        status = OK if proc.returncode == 0 else RUNTIME_ERROR
    except subprocess.TimeoutExpired:
        # The group may already be gone if the program ended at the deadline.
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
        out, err = proc.communicate(timeout=5)
        status = TIMEOUT
    elapsed = time.monotonic() - t0
    if len(out) > C.MAX_OUTPUT_BYTES:
        status, out = OUTPUT_LIMIT, out[:C.MAX_OUTPUT_BYTES]
    return RunResult(status, out, err[:4000], proc.returncode, elapsed)


def run_program(prog: Program, stdin_data: str, timeout: float = C.RUN_TIMEOUT) -> RunResult:
    return run_process(prog.argv, stdin_data=stdin_data, timeout=timeout, cwd=prog.workdir)


# ----------------------------------------------------------- compilation ---
# Some older solutions declare globals that collide with C++17's std::, hence the fallback.
CPP_STDS = ["gnu++17", "gnu++14", "gnu++11"]
CPP_FLAGS = ["-O1", "-w", "-pipe"]
JAVA_MEM = ["-Xmx1024m", "-Xss256m", "-XX:+UseSerialGC", "-XX:TieredStopAtLevel=1"]
# JVM warnings go to stderr, never into the program's output.
JAVA_QUIET = ["-XX:-UsePerfData", "-Xlog:disable", "-Xlog:all=warning:stderr"]


def _cpp_argv(binary: Path, stack_mb: int | None) -> list[str]:
    """The run command under `ulimit -v`, plus `ulimit -s` when `stack_mb` is set."""
    limits = f"ulimit -v {C.RUN_MEMORY_MB * 1024}"
    if stack_mb:
        limits += f"; ulimit -s {stack_mb * 1024}"
    return ["/bin/bash", "-c", f"{limits}; exec \"$0\"", str(binary)]


def _compile_cpp(workdir: Path, source: str, coverage: bool, std_hint: str | None,
                 stack_mb: int | None) -> tuple[Program | None, str]:
    src, obj, binary = workdir / "main.cpp", workdir / "main.o", workdir / "prog"
    src.write_text(source, encoding="utf-8")
    for cpp_std in [std_hint] if std_hint else CPP_STDS:
        if coverage:
            # Separate compile and link, so gcov finds main.gcno.
            res = run_process(["g++", f"-std={cpp_std}", "-w", "-pipe", "--coverage",
                               "-fprofile-abs-path", "-c", str(src), "-o", str(obj)],
                              timeout=C.COMPILE_TIMEOUT, cwd=workdir)
            if res.status == OK:
                res = run_process(["g++", "--coverage", str(obj), "-o", str(binary)],
                                  timeout=C.COMPILE_TIMEOUT, cwd=workdir)
        else:
            res = run_process(["g++", f"-std={cpp_std}", *CPP_FLAGS, str(src), "-o", str(binary)],
                              timeout=C.COMPILE_TIMEOUT, cwd=workdir)
        if res.status == OK and binary.exists():
            return Program(workdir, _cpp_argv(binary, stack_mb)), ""
    return None, res.stderr or res.stdout


def _compile_java(workdir: Path, source: str) -> tuple[Program | None, str]:
    cls = _java_class(source)
    src = workdir / f"{cls}.java"
    src.write_text(source, encoding="utf-8")
    res = run_process(["javac", "-nowarn", "-encoding", "UTF-8", "-d", str(workdir), str(src)],
                      timeout=C.COMPILE_TIMEOUT, cwd=workdir)
    if res.status != OK or not (workdir / f"{cls}.class").exists():
        return None, res.stderr or res.stdout
    return Program(workdir, ["java", *JAVA_MEM, *JAVA_QUIET, "-cp", str(workdir), cls]), ""


def compile_program(lang: str, source: str, *, coverage: bool = False,
                    std_hint: str | None = None,
                    stack_mb: int | None = None) -> tuple[Program | None, str]:
    """(program, "") in a fresh directory the caller cleans up, or (None, compiler output)."""
    workdir = Path(tempfile.mkdtemp(prefix="job_", dir=C.WORK_DIR))
    if lang == "cpp":
        prog, err = _compile_cpp(workdir, source, coverage, std_hint, stack_mb)
    else:
        prog, err = _compile_java(workdir, source)
    if prog is None:
        shutil.rmtree(workdir, ignore_errors=True)
    return prog, err


_PUBLIC_CLASS = re.compile(
    r"^[ \t]*public[ \t]+(?:final[ \t]+|abstract[ \t]+)?class[ \t]+(\w+)", re.MULTILINE)
_ANY_CLASS = re.compile(
    r"^[ \t]*(?:final[ \t]+|abstract[ \t]+)?class[ \t]+(\w+)", re.MULTILINE)


def _java_class(source: str) -> str:
    """The public class, else `Main` if declared or no class is, else the first class."""
    if m := _PUBLIC_CLASS.search(source):
        return m.group(1)
    names = _ANY_CLASS.findall(source)
    return names[0] if names and "Main" not in names else "Main"


# ------------------------------------------------------ output comparison ---
_NUM = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def normalize(out: str) -> str:
    """Unify line endings and drop trailing whitespace and blank lines."""
    lines = [ln.rstrip() for ln in out.replace("\r\n", "\n").split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def outputs_match(a: str, b: str, *, tol: float = 1e-6) -> bool:
    """Equal after `normalize`, or token by token with numbers equal to within `tol`,
    relative or absolute."""
    na, nb = normalize(a), normalize(b)
    if na == nb:
        return True
    ta, tb = na.split(), nb.split()
    return len(ta) == len(tb) and all(
        x == y or (_NUM.match(x) and _NUM.match(y)
                   and math.isclose(float(x), float(y), rel_tol=tol, abs_tol=tol))
        for x, y in zip(ta, tb))
