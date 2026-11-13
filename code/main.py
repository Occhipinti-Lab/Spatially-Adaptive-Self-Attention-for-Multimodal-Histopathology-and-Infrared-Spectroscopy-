

from __future__ import annotations

if __package__ in {None, ""}:
    # Direct-file execution places this folder, rather than its parent, on
    # sys.path. Resolve the package root without changing the working directory.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runner import main


if __name__ == "__main__":
    main()

