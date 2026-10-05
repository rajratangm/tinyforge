"""Structured warnings shared by CLI, API and UI.

Every diagnostic carries a stable code so the UI can render it and CI can assert on it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class Level(str, Enum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


@dataclass
class Diagnostic:
    code: str
    level: Level
    message: str
    fix: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["level"] = self.level.value
        return d


@dataclass
class Report:
    items: list[Diagnostic] = field(default_factory=list)

    def add(self, code: str, level: Level, message: str, fix: str = "") -> None:
        self.items.append(Diagnostic(code, level, message, fix))

    def extend(self, other: Report) -> None:
        self.items.extend(other.items)

    @property
    def has_errors(self) -> bool:
        return any(i.level is Level.ERROR for i in self.items)

    def to_list(self) -> list[dict]:
        return [i.to_dict() for i in self.items]
