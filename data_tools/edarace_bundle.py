"""Prepare a source ZIP whose Docker build never downloads dependencies."""

from __future__ import annotations

import argparse
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def wheel_metadata(path: Path) -> tuple[str, str, list[str]]:
    with zipfile.ZipFile(path) as archive:
        metadata = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        wheels = [n for n in archive.namelist() if n.endswith(".dist-info/WHEEL")]
        if len(metadata) != 1 or len(wheels) != 1:
            raise ValueError(f"invalid wheel metadata: {path.name}")
        info = BytesParser().parsebytes(archive.read(metadata[0]))
        wheel = BytesParser().parsebytes(archive.read(wheels[0]))
        return normalized(info["Name"]), info["Version"], wheel.get_all("Tag", [])


def requirements(path: Path) -> dict[str, str]:
    result = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+-]+)", line)
        if match is None:
            raise ValueError(f"offline dependencies must be pinned: {line}")
        name, version = normalized(match[1]), match[2]
        if name in result or name.startswith(("torch", "nvidia")):
            raise ValueError(f"duplicate or official-runtime dependency: {name}")
        result[name] = version
    return result


def prepare(workspace: Path, wheelhouse: Path, output: Path, budget: float) -> dict:
    workspace, wheelhouse, output = workspace.resolve(), wheelhouse.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle: {output}")
    if not 0 < budget <= 7200:
        raise ValueError("batch budget must be positive and at most 7200 seconds")
    expected = requirements(workspace / "requirements.edarace.txt")
    assets = {}
    for path in sorted(wheelhouse.glob("*.whl")):
        name, version, tags = wheel_metadata(path)
        if name not in expected or version != expected[name] or name in assets:
            raise ValueError(f"unexpected or duplicate offline dependency: {path.name}")
        compatible = any(
            tag in {"py3-none-any", "py2.py3-none-any"}
            or ("manylinux" in tag and tag.endswith("x86_64") and (tag.startswith("cp312-") or "-abi3-" in tag))
            for tag in tags
        )
        if not compatible:
            raise ValueError(f"wheel does not target Python 3.12 / Linux x86_64: {path.name}")
        url = f"https://pypi.org/pypi/{name}/{version}/json"
        with urllib.request.urlopen(url, timeout=30) as response:
            release = json.load(response)
        files = [item for item in release["urls"] if item["filename"] == path.name]
        digest = sha256(path)
        if len(files) != 1 or digest != files[0]["digests"]["sha256"]:
            raise ValueError(f"wheel does not match official PyPI SHA256: {path.name}")
        assets[name] = {"path": path, "filename": path.name, "bytes": path.stat().st_size, "sha256": digest, "name": name, "version": version, "source": files[0]["url"]}
        print(f"Verified PyPI wheel: {path.name}", flush=True)
    if assets.keys() != expected.keys():
        raise ValueError(f"missing offline dependencies: {sorted(expected.keys() - assets.keys())}")

    lock_path = workspace / "models/models.lock.json"
    lock_bytes = lock_path.read_bytes()
    lock = json.loads(lock_bytes)
    if {entry["task"] for entry in lock["models"]} != {"component", "ocr", "pin", "wire"}:
        raise ValueError("expected canonical four-task production model lock")
    locked_models = set()
    for entry in lock["models"]:
        model = (lock_path.parent / entry["path"]).resolve()
        model.relative_to((workspace / "models/releases/v8-portable").resolve())
        if not model.is_file() or sha256(model) != entry["sha256"]:
            raise ValueError(f"production model missing or checksum mismatch: {entry['task']}")
        locked_models.add(model)

    output.mkdir(parents=True)
    package = output / "package"
    package.mkdir()
    build_source = output / "wheel-source"
    build_source.mkdir()
    for name in ("pyproject.toml", "README.md"):
        shutil.copy2(workspace / name, build_source / name)
    shutil.copytree(workspace / "src", build_source / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info", "evaluation"))
    wheels = package / "wheels"
    wheels.mkdir()
    env = dict(os.environ, PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
    command = [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "--no-index", "--wheel-dir", str(wheels), str(build_source)]
    result = subprocess.run(command, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    (output / "project-wheel-build.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"offline project wheel build failed; see {output / 'project-wheel-build.log'}")
    project = wheels / "pcb_schematic_parser-0.1.0-py3-none-any.whl"
    if not project.is_file():
        raise ValueError("expected prebuilt application wheel missing")
    with zipfile.ZipFile(project) as archive:
        if any(name.startswith("pcbparse/evaluation/") for name in archive.namelist()):
            raise ValueError("application wheel contains offline evaluation code")
    entries = []
    for row in assets.values():
        shutil.copy2(row["path"], wheels / row["filename"])
        entries.append({key: value for key, value in row.items() if key != "path"})
    entries.append({"filename": project.name, "bytes": project.stat().st_size, "sha256": sha256(project), "name": "pcb-schematic-parser", "version": "0.1.0", "source": "current workspace source snapshot"})
    manifest = {"schema_version": 1, "target": "cp312-linux-x86_64", "project_wheel": project.name, "files": sorted(entries, key=lambda row: row["filename"])}
    (wheels / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(workspace / "requirements.edarace.txt", package / "requirements.edarace.txt")
    shutil.copy2(workspace / "docker/Dockerfile", package / "Dockerfile")
    shutil.copy2(workspace / "docker/agent_system", package / "run.sh")
    (package / "docker").mkdir()
    for name in ("agent_system", "verify_wheelhouse.py"):
        shutil.copy2(workspace / "docker" / name, package / "docker" / name)
    (package / ".dockerignore").write_text("__pycache__/\n*.pyc\n.git/\n", encoding="utf-8")
    (package / "configs/profiles").mkdir(parents=True)
    for name in ("pipeline.edarace.yaml", "official_schema.yaml", "pipeline.yaml"):
        shutil.copy2(workspace / "configs" / name, package / "configs" / name)
    for name in ("competition.yaml", "baseline.yaml"):
        shutil.copy2(workspace / "configs/profiles" / name, package / "configs/profiles" / name)
    config_path = package / "configs/pipeline.edarace.yaml"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["run"]["max_total_seconds"] = budget
    if config.get("qwen", {}).get("enabled"):
        raise ValueError("expected current production profile with API use disabled")
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    def unused_model_assets(directory: str, names: list[str]) -> list[str]:
        skipped = []
        for name in names:
            path = Path(directory) / name
            if name == "__pycache__" or path.suffix == ".pyc":
                skipped.append(name)
            elif path.suffix in {".pt", ".pth", ".onnx", ".ts"} and path.resolve() not in locked_models:
                skipped.append(name)
        return skipped

    shutil.copytree(workspace / "models/releases/v8-portable", package / "models/releases/v8-portable", ignore=unused_model_assets)
    (package / "models/models.lock.json").write_bytes(lock_bytes)
    (package / "SUBMISSION_README.txt").write_text(
        "PCBParse offline EDARace source submission\n"
        "Upload ZIP directly. Dockerfile and run.sh are at archive root.\n"
        "Dependencies and application wheel are bundled and hash-verified.\n"
        "Docker build uses no package index, no dependency download, and no source build.\n"
        "Entry point: /submission/run.sh -input /input -output /output\n"
        "Predictions: /output/final_results/<case_id>_Pred.json\n"
        "Decision log: /output/decision_log.json\n"
        "Canonical production models and their lock are unchanged.\n"
        f"Batch time budget: {budget:g} seconds.\n"
        "Build and inference verification evidence is delivered separately.\n",
        encoding="utf-8",
    )
    records = [{"path": path.relative_to(package).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(package.rglob("*")) if path.is_file()]
    archive_path = output / "pcbparse_edarace_offline.zip"
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for row in records:
            info = zipfile.ZipInfo(row["path"], date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (0o100555 if row["path"] in {"run.sh", "docker/agent_system"} else 0o100644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, (package / row["path"]).read_bytes(), compresslevel=1)
    report = {"kind": "edarace-offline-source-bundle", "archive": str(archive_path), "bytes": archive_path.stat().st_size, "sha256": sha256(archive_path), "files": records, "model_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(), "batch_budget_seconds": budget, "offline_wheels": len(entries), "docker_build_verified": False, "inference_verified": False}
    (output / "package-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "pcbparse_edarace_offline.zip.sha256").write_text(f"{report['sha256']}  {archive_path.name}\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-total-seconds", type=float, default=4800.0)
    args = parser.parse_args()
    report = prepare(args.workspace_root, args.wheelhouse, args.output, args.max_total_seconds)
    print(json.dumps({key: value for key, value in report.items() if key != "files"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
