"""Read a folder of notes (an Obsidian vault, or any Markdown folder) as clean documents.

Hidden folders (.obsidian, .git, .claude, ...) and excluded folders are skipped.
Frontmatter, %% comments %% and embeds are dropped; [[links]] become their text.
"""

import os
import re
from collections.abc import Iterable
from pathlib import Path

from .rag import Document

SUFFIXES = {".md", ".txt"}

_FRONTMATTER = re.compile(r"\A---[ \t]*\n.*?\n---[ \t]*(?:\n|\Z)", re.S)
_COMMENT = re.compile(r"%%.*?%%", re.S)
_EMBED = re.compile(r"!\[\[[^\]]*\]\]")
_WIKILINK = re.compile(r"\[\[([^\]|#]*)(?:#[^\]|]*)?(?:\|([^\]]*))?\]\]")


def clean_markdown(text: str) -> str:
    text = text.lstrip("﻿")
    text = _FRONTMATTER.sub("", text, count=1)
    text = _COMMENT.sub("", text)
    text = _EMBED.sub("", text)
    text = _WIKILINK.sub(_link_text, text)
    return text.strip()


def _link_text(m: re.Match) -> str:
    alias, target = m.group(2), m.group(1)
    return (alias or target.split("/")[-1]).strip()


def parse_exclude(value: str) -> set[str]:
    """'7 - Dev Folder, 6 - Brian/Private' -> {'7 - Dev Folder', '6 - Brian/Private'}"""
    return {p.strip().strip("/") for p in value.split(",") if p.strip().strip("/")}


def find_note_files(root: Path, exclude: Iterable[str] = ()) -> list[Path]:
    excluded = set(exclude)
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        dirnames[:] = sorted(
            d for d in dirnames if not d.startswith(".") and (rel / d).as_posix() not in excluded
        )
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if (
                not name.startswith(".")
                and path.suffix.lower() in SUFFIXES
                and (rel / name).as_posix() not in excluded
            ):
                found.append(path)
    return found


def read_notes(root: Path, exclude: Iterable[str] = ()) -> tuple[list[Document], list[str]]:
    """Every note under root as a Document, plus the files that could not be read."""
    docs: list[Document] = []
    skipped: list[str] = []
    for path in find_note_files(root, exclude):
        source = path.relative_to(root).as_posix()
        try:
            raw = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as e:
            skipped.append(f"{source}: {type(e).__name__}")
            continue
        folder = path.parent.relative_to(root).as_posix()
        heading = path.stem if folder == "." else f"{path.stem} ({folder})"
        docs.append(Document(source=source, text=clean_markdown(raw), heading=heading))
    return docs, skipped
