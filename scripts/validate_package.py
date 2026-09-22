"""Validate the contents of a built Blender add-on archive."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZipFile


REQUIRED_FILES = {
    "gaussian_camera_rig/__init__.py",
    "gaussian_camera_rig/raycast_export.py",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()

    with ZipFile(args.archive) as archive:
        names = set(archive.namelist())

    missing = REQUIRED_FILES - names
    forbidden = {
        name for name in names
        if "__pycache__/" in name or name.endswith((".pyc", ".pyo"))
    }
    if missing or forbidden:
        if missing:
            print(f"Missing required archive entries: {sorted(missing)}")
        if forbidden:
            print(f"Forbidden generated entries: {sorted(forbidden)}")
        return 1

    print(f"Validated {args.archive} ({len(names)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
