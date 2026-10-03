"""Source-level mutants from `universalmutator`, drawn round-robin across operator
families so that no single operator dominates a solution's mutants."""
from __future__ import annotations

import contextlib
import io
import random
import threading
from dataclasses import dataclass

from universalmutator import mutator as _um

from pipeline import config as C
from pipeline import exec_harness as EX

_RULES = {
    "cpp": ["universal.rules", "cpp.rules", "c_like.rules"],
    "java": ["universal.rules", "java.rules", "c_like.rules"],
}


@dataclass
class Mutant:
    mutant_id: str
    operator: str
    line: int
    source: str


# Checked in order against a rule's left-hand side.
_FAMILY_TOKENS = [
    ("relational", ("==", "!=", "<=", ">=", "<", ">")),
    ("logical", ("&&", "||", "!")),
    ("assignment", ("+=", "-=", "*=", "/=", "%=")),
    ("unary", ("++", "--")),
    ("bitwise", ("<<", ">>", "&", "|", "^", "~")),
    ("arithmetic", ("+", "-", "*", "/", "%")),
]


def _family(rule_text: str) -> str:
    """The classic operator family of a universalmutator rule ("<lhs> ==> <rhs>")."""
    lhs, _, rhs = (part.strip() for part in rule_text.partition("==>"))
    if not rhs:
        return "deletion"
    if any(t in lhs for t in ("true", "false")):
        return "return_value"
    if '"' in lhs or "\\d" in lhs:
        return "constant"
    for name, toks in _FAMILY_TOKENS:
        if any(t in lhs for t in toks):
            return name
    return "other"


_STDOUT_LOCK = threading.Lock()


def generate_mutants(source: str, lang: str, *, limit: int) -> list[Mutant]:
    """Up to `limit` single-point mutants, spread across operator families."""
    lines = source.split("\n")
    # The tool prints to stdout; the lock keeps pool threads from overlapping redirects.
    with _STDOUT_LOCK, contextlib.redirect_stdout(io.StringIO()):
        raw = _um.mutants_regexp(source=lines, ruleFiles=_RULES[lang], ignorePatterns=[])

    by_family: dict[str, list] = {}
    for lineno, mline, rule_used, _ in raw:
        by_family.setdefault(_family(rule_used[0]), []).append((lineno, mline))

    rng = random.Random(0)
    for lst in by_family.values():
        rng.shuffle(lst)
    families = sorted(by_family)
    mutants, seen, idx = [], set(), 0
    while len(mutants) < limit and any(by_family.values()):
        fam = families[idx % len(families)]
        idx += 1
        if not by_family[fam]:
            continue
        lineno, mline = by_family[fam].pop()
        mutated = "\n".join(lines[:lineno - 1] + [mline.rstrip("\n")] + lines[lineno:])
        if mutated == source or mutated in seen:
            continue
        seen.add(mutated)
        mutants.append(Mutant(f"m{len(mutants):03d}", fam, lineno, mutated))
    return mutants


def mutant_timeout(orig_max: float) -> float:
    """A mutant's run limit, scaled off the original's slowest run."""
    return max(2.0, min(C.RUN_TIMEOUT, 3.0 * orig_max + 1.0))


def kills(mp: EX.Program, cases: list[dict], timeout: float) -> bool:
    """Whether the mutant crashes, times out, or changes the output on any case."""
    for e in cases:
        r = EX.run_program(mp, e["input"], timeout=timeout)
        if r.status != EX.OK or not EX.outputs_match(r.stdout, e["output"]):
            return True
    return False
