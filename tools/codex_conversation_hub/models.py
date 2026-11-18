"""Small data structures shared by the scanner, storage layer, and web UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceConfig:
    """A user-configured local source tree."""

    id: str
    kind: str
    path: str
    enabled: bool = True


@dataclass
class HubConfig:
    """Runtime configuration kept outside the indexed source data."""

    data_dir: Path
    archive_originals: bool = True
    max_file_bytes: int = 250_000_000
    preview_chars: int = 12_000
    sources: list[SourceConfig] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "data_dir": str(self.data_dir),
            "archive_originals": self.archive_originals,
            "max_file_bytes": self.max_file_bytes,
            "preview_chars": self.preview_chars,
            "sources": [
                {
                    "id": source.id,
                    "kind": source.kind,
                    "path": source.path,
                    "enabled": source.enabled,
                }
                for source in self.sources
            ],
        }
