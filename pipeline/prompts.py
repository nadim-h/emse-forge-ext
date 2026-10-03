"""Task prompts from `prompts/<task>.md`: the system message, a `---` line, then
the user template with `$name` placeholders."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from string import Template

from pipeline import config as C


@dataclass(frozen=True)
class Prompt:
    system: str
    template: Template

    def messages(self, **fields) -> list[dict]:
        return [{"role": "system", "content": self.system},
                {"role": "user", "content": self.template.substitute(**fields)}]


@cache
def load(name: str) -> Prompt:
    text = (C.ROOT / "prompts" / f"{name}.md").read_text(encoding="utf-8")
    head, _, body = text.partition("\n---\n")
    return Prompt(system=head.strip(), template=Template(body.strip()))
