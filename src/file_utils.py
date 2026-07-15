from __future__ import annotations

import os


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)
