"""Resumable job runners: `run_parallel` (thread pool, for subprocess work) and
`run_llm` (asyncio, for the model endpoint).

Each result is stored with `_key = str(key_of(job))`; a job that raises is stored
as `{"error": "harness: ..."}`. Rows passing `keep` are checkpointed to
`<out>.partial.jsonl` and reused on resume.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    """All rows of a jsonl file; [] if missing. A line cut off by a kill is skipped."""
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """Write atomically via a temp file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fo:
        fo.writelines(json.dumps(r) + "\n" for r in rows)
    tmp.replace(path)


class _Run:
    """The cached rows of one output file, and the checkpoint new rows go to."""

    def __init__(self, out_path: Path, key_of: Callable, resume: bool, keep: Callable):
        self.out_path, self.key_of, self.keep = out_path, key_of, keep
        self.ckpt = out_path.with_suffix(".partial.jsonl")
        self.rows = {}
        if resume:
            self.rows = {r["_key"]: r for r in read_jsonl(out_path) + read_jsonl(self.ckpt)
                         if keep(r)}

    def pending(self, jobs: list) -> list:
        return [j for j in jobs if str(self.key_of(j)) not in self.rows]

    def record(self, ck, job, r: dict) -> None:
        """Store a job's row, and checkpoint it if it is kept."""
        r["_key"] = str(self.key_of(job))
        self.rows[r["_key"]] = r
        if self.keep(r):
            ck.write(json.dumps(r) + "\n")
            ck.flush()

    def finish(self) -> list[dict]:
        """Write every row sorted by key and drop the checkpoint."""
        rows = sorted(self.rows.values(), key=lambda r: r["_key"])
        write_jsonl(self.out_path, rows)
        self.ckpt.unlink(missing_ok=True)
        return rows


def run_parallel(jobs: list, fn: Callable, *, key_of: Callable, out_path: Path,
                 workers: int = 24, resume: bool = True,
                 keep: Callable[[dict], bool] = lambda r: True) -> list[dict]:
    """Map `fn` over `jobs` in a thread pool."""
    run = _Run(out_path, key_of, resume, keep)
    with run.ckpt.open("a", encoding="utf-8") as ck, ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fn, j): j for j in run.pending(jobs)}
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception as e:                           # noqa: BLE001
                r = {"error": f"harness: {e}"}
            run.record(ck, futs[f], r)
    return run.finish()


async def run_llm(jobs: list, coro: Callable, *, key_of: Callable, out_path: Path,
                  concurrency: int = 64, resume: bool = True,
                  keep: Callable[[dict], bool] = lambda r: True) -> list[dict]:
    """Same contract as `run_parallel`, for an async per-job coroutine."""
    run = _Run(out_path, key_of, resume, keep)
    sem = asyncio.Semaphore(concurrency)

    async def one(job, ck):
        async with sem:
            try:
                r = await coro(job)
            except Exception as e:                           # noqa: BLE001
                r = {"error": f"harness: {e}"}
            run.record(ck, job, r)

    with run.ckpt.open("a", encoding="utf-8") as ck:
        await asyncio.gather(*(one(j, ck) for j in run.pending(jobs)))
    return run.finish()
