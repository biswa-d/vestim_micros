#!/usr/bin/env python3
"""Copy only inference assets; preserve the development/source directories."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "share_bundle"))
    parser.add_argument("--test-dir", default=str(Path(__file__).resolve().parent / "LG_NMC_test_data"))
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory; preserving existing files: {output}")
    files = []
    for name in ["run_offline_inference.py", "ecm_inference.py", "requirements.txt",
                 "run_test.bat", "run_test.sh", "README.md", "ECM_README.md"]:
        files.append((source / name, Path(name)))
    for job in sorted(source.glob("job_*")):
        if not job.is_dir():
            continue
        model_dirs = [p.parent for p in job.glob("models/**/task_info.json")
                      if (p.parent / "best_model.pth").is_file() or (p.parent / "best_model_export.pt").is_file()]
        if len(model_dirs) != 1:
            raise ValueError(f"Keep one selected trained task in {job.name} before packaging")
        model = model_dirs[0]
        metadata = json.loads((job / "job_metadata.json").read_text(encoding="utf-8"))
        selected = [job / "job_metadata.json", model / "task_info.json"]
        if (job / "augmentation_metadata.json").is_file():
            selected.append(job / "augmentation_metadata.json")
        checkpoint = model / "best_model.pth"
        selected.append(checkpoint if checkpoint.is_file() else model / "best_model_export.pt")
        if metadata.get("normalization_applied"):
            relative = Path(str(metadata.get("scaler_path", "scalers/augmentation_scaler.joblib")).replace(chr(92), "/"))
            scaler = job / relative
            if relative.is_absolute() or ":" in str(relative) or ".." in relative.parts or not scaler.is_file():
                scaler = job / "scalers" / relative.name
            selected.append(scaler)
        files.extend((p, p.relative_to(source)) for p in selected)
    for config in sorted(source.glob("ecm_*/ecm_config.json")):
        files.append((config, config.relative_to(source)))
        for entry in json.loads(config.read_text(encoding="utf-8"))["parameters"]:
            path = (config.parent / entry["path"]).resolve()
            if not path.is_relative_to(config.parent.resolve()):
                raise ValueError("ECM parameter path escapes its model directory")
            files.append((path, path.relative_to(source)))
    test_dir = Path(args.test_dir).expanduser().resolve()
    tests = sorted(test_dir.glob("*.csv"))
    if not tests:
        raise FileNotFoundError(f"No raw test CSVs found in {test_dir}")
    files.extend((p, Path("LG_NMC_test_data") / p.name) for p in tests)
    for original, relative in files:
        if not original.is_file():
            raise FileNotFoundError(original)
        if not (output / relative).resolve().is_relative_to(output):
            raise ValueError("Invalid bundle destination")
    output.mkdir(parents=True)
    manifest = []
    for original, relative in files:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
        with destination.open("rb") as fh:
            digest = hashlib.file_digest(fh, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(fh.read()).hexdigest()
        manifest.append(dict(path=relative.as_posix(), bytes=destination.stat().st_size, sha256=digest))
    (output / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Created {output}: {len(files)} files, {sum(p['bytes'] for p in manifest):,} bytes")
    print("MATLAB scripts, benchmark data, training logs, and cached predictions were excluded.")


if __name__ == "__main__":
    main()
