"""Verify offline installation assets before pip or application code runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


def verify(root: Path) -> int:
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("target") != "cp312-linux-x86_64":
        raise ValueError("unsupported wheelhouse manifest")
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("wheelhouse manifest must list installation files")
    expected = set()
    for row in rows:
        name = row["filename"]
        if Path(name).name != name or not name.endswith(".whl") or name in expected:
            raise ValueError(f"invalid or duplicate wheel filename: {name}")
        if name.lower().startswith(("torch-", "torchvision-", "torchaudio-", "nvidia_", "nvidia-")):
            raise ValueError(f"official CUDA runtime must not be replaced: {name}")
        expected.add(name)
        path = root / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != row["bytes"]:
            raise ValueError(f"wheel missing or size mismatch: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != row["sha256"]:
            raise ValueError(f"wheel checksum mismatch: {name}")
    actual = {path.name for path in root.iterdir() if path.is_file()}
    if actual != expected | {"manifest.json"}:
        raise ValueError("wheelhouse contains missing or unlisted files")
    project = manifest.get("project_wheel")
    if project != "pcb_schematic_parser-0.1.0-py3-none-any.whl" or project not in expected:
        raise ValueError("expected application wheel missing")
    print(f"Verified {len(expected)} offline wheels", flush=True)
    return len(expected)


if __name__ == "__main__":
    verify(Path(sys.argv[1]))
