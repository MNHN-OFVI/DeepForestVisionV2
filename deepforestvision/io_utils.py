from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple


def ensure_empty_dir(path: Path) -> None:
    """
    Create directory if missing and remove all its contents if it exists.
    """
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)


def list_media_files(root: Path, extensions: Tuple[str, ...]) -> List[Path]:
    """
    Recursively list files under root with one of the given extensions.
    Returned paths are absolute Paths.
    """
    out: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix in extensions:
                out.append(p)
    return sorted(out)


def safe_copy(src: Path, dst: Path) -> None:
    """
    Copy a file and create parent dirs if needed.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def stem_name(p: Path) -> str:
    """
    Filename without directory (keeps suffix if needed elsewhere).
    """
    return p.name