"""Build a deterministic Blender add-on archive.

Usage:
    python scripts/build_addon_zip.py
    python scripts/build_addon_zip.py --output dist/gaussian_camera_rig.zip
"""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "gaussian_camera_rig"
DEFAULT_OUTPUT = ROOT / "gaussian_camera_rig.zip"
IGNORED_NAMES = {"__pycache__", ".DS_Store", "Thumbs.db"}


def package_files():
    for path in sorted(PACKAGE.rglob("*")):
        if not path.is_file():
            continue
        if any(part in IGNORED_NAMES for part in path.parts):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        yield path, path.relative_to(ROOT).as_posix()


def build(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for source, name in package_files():
            info = ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    build(output)
    entries = [name for _, name in package_files()]
    print(f"Wrote {output} ({len(entries)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
