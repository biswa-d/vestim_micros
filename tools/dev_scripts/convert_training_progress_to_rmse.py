"""Convert normalized training-progress losses to original-scale RMSE.

The training progress CSV stores MSE values in normalized target space. This
script converts those MSE columns to the same display units used by the GUI
when only the CSV and scaler are available offline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd


LOSS_COLUMNS = ("train_loss_norm", "val_loss_norm", "best_val_loss_norm")


def _unit_info(target_column: str) -> tuple[str, float]:
    target = target_column.lower()
    if "voltage" in target:
        return "mV", 1000.0
    if "soc" in target or "soe" in target or "sop" in target:
        return "%", 100.0
    if "temperature" in target or "temp" in target:
        return "degC", 1.0
    return "original_units", 1.0


def _candidate_target_names(target_column: str) -> list[str]:
    target = target_column.strip()
    target_lower = target.lower()
    candidates = [target]

    if "voltage" in target_lower:
        candidates.extend(["Voltage(V)", "voltage(v)", "Voltage", "voltage", "VOLTAGE"])
    elif "soc" in target_lower:
        candidates.extend(["SOC", "soc", "Soc"])
    elif "temperature" in target_lower or "temp" in target_lower:
        candidates.extend(
            [
                "Temperature (C)",
                "temperature (c)",
                "Temperature(C)",
                "temperature(c)",
                "Temperature",
                "temperature",
                "TEMPERATURE",
                "Temp",
                "temp",
            ]
        )

    seen = set()
    return [name for name in candidates if not (name.lower() in seen or seen.add(name.lower()))]


def _find_training_csv(path: Path) -> Path:
    if path.is_file():
        return path

    candidates = [
        path / "logs" / "training_progress.csv",
        path / "training_progress.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find training_progress.csv in {path} or {path / 'logs'}"
    )


def _walk_parents(start: Path) -> Iterable[Path]:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    yield current
    yield from current.parents


def _find_job_folder(training_csv: Path, explicit_job_dir: str | None) -> Path | None:
    if explicit_job_dir:
        job_dir = Path(explicit_job_dir).expanduser().resolve()
        if not job_dir.exists():
            raise FileNotFoundError(f"Job directory does not exist: {job_dir}")
        return job_dir

    for parent in _walk_parents(training_csv):
        if (parent / "job_metadata.json").exists() or (parent / "scalers").exists():
            return parent
    return None


def _load_scaler(job_dir: Path | None, explicit_scaler: str | None):
    if explicit_scaler:
        scaler_path = Path(explicit_scaler).expanduser().resolve()
    elif job_dir is not None:
        scaler_path = job_dir / "scalers" / "augmentation_scaler.joblib"
    else:
        return None, None

    if not scaler_path.exists():
        return None, scaler_path
    return joblib.load(scaler_path), scaler_path


def _load_normalized_columns(job_dir: Path | None, scaler) -> list[str]:
    if job_dir is not None:
        metadata_path = job_dir / "job_metadata.json"
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            columns = metadata.get("normalized_columns") or []
            if columns:
                return list(columns)

    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)

    return []


def _target_index(scaler, normalized_columns: list[str], target_column: str) -> int:
    feature_names = []
    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        feature_names = [str(name) for name in scaler.feature_names_in_]
    elif normalized_columns:
        feature_names = [str(name) for name in normalized_columns]

    if not feature_names:
        raise ValueError("Scaler does not expose feature names and metadata has no normalized_columns.")

    lookup = {name.strip().lower(): idx for idx, name in enumerate(feature_names)}
    for candidate in _candidate_target_names(target_column):
        idx = lookup.get(candidate.strip().lower())
        if idx is not None:
            return idx

    raise ValueError(
        f"Target column '{target_column}' was not found in scaler features: {feature_names}"
    )


def _rmse_scale_factor(scaler, target_idx: int) -> tuple[float, dict[str, float]]:
    if scaler is None:
        return 1.0, {}

    if hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
        data_min = float(np.array(scaler.data_min_).flatten()[target_idx])
        data_max = float(np.array(scaler.data_max_).flatten()[target_idx])
        return data_max - data_min, {"target_min": data_min, "target_max": data_max}

    if hasattr(scaler, "scale_") and hasattr(scaler, "mean_"):
        std = float(np.array(scaler.scale_).flatten()[target_idx])
        mean = float(np.array(scaler.mean_).flatten()[target_idx])
        return std, {"target_mean": mean, "target_std": std}

    raise ValueError(f"Unsupported scaler type for RMSE conversion: {type(scaler).__name__}")


def _loss_to_rmse(loss_value, rmse_factor: float, unit_multiplier: float):
    if pd.isna(loss_value):
        return np.nan
    loss = float(loss_value)
    if loss < 0:
        return np.nan
    return math.sqrt(loss) * rmse_factor * unit_multiplier


def convert_training_progress(
    path: str,
    target_column: str,
    job_dir: str | None = None,
    scaler_path: str | None = None,
    output: str | None = None,
) -> Path:
    training_csv = _find_training_csv(Path(path).expanduser().resolve())
    resolved_job_dir = _find_job_folder(training_csv, job_dir)
    scaler, resolved_scaler_path = _load_scaler(resolved_job_dir, scaler_path)
    normalized_columns = _load_normalized_columns(resolved_job_dir, scaler)

    target_idx = _target_index(scaler, normalized_columns, target_column) if scaler is not None else -1
    rmse_factor, scaler_stats = _rmse_scale_factor(scaler, target_idx) if scaler is not None else (1.0, {})
    unit, unit_multiplier = _unit_info(target_column)

    df = pd.read_csv(training_csv, comment="#", sep=None, engine="python")
    for column in LOSS_COLUMNS:
        if column in df.columns:
            output_column = column.replace("_loss_norm", f"_rmse_{unit}")
            df[output_column] = df[column].apply(
                lambda value: _loss_to_rmse(value, rmse_factor, unit_multiplier)
            )

    df["rmse_unit"] = unit
    df["rmse_conversion_factor"] = rmse_factor * unit_multiplier
    df["target_column_for_conversion"] = target_column
    for key, value in scaler_stats.items():
        df[key] = value

    if output:
        output_path = Path(output).expanduser().resolve()
    else:
        output_path = training_csv.with_name(training_csv.stem + f"_rmse_{unit}.csv")

    df.to_csv(output_path, index=False)

    print(f"Read: {training_csv}")
    if resolved_scaler_path:
        print(f"Scaler: {resolved_scaler_path}")
    if resolved_job_dir:
        print(f"Job folder: {resolved_job_dir}")
    print(f"Target: {target_column}")
    print(f"RMSE conversion factor: {rmse_factor * unit_multiplier:.10g} {unit} per normalized RMSE")
    print(f"Wrote: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert training_progress.csv normalized MSE columns to original-scale RMSE."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to training_progress.csv or a model directory containing logs/training_progress.csv.",
    )
    parser.add_argument(
        "--progress-csv",
        dest="progress_csv",
        help="Path to training_progress.csv or a model directory containing logs/training_progress.csv.",
    )
    parser.add_argument(
        "--target-column",
        default="Voltage(V)",
        help="Target column used for training, e.g. Voltage(V), SOC, Temperature(C).",
    )
    parser.add_argument("--job-dir", help="Job folder containing job_metadata.json and scalers/.")
    parser.add_argument(
        "--scaler",
        dest="scaler",
        help="Explicit path to augmentation_scaler.joblib.",
    )
    parser.add_argument(
        "--scaler-path",
        dest="scaler",
        help="Explicit path to augmentation_scaler.joblib.",
    )
    parser.add_argument("--output", help="Output CSV path. Defaults beside training_progress.csv.")
    args = parser.parse_args()

    selected_path = args.progress_csv or args.path
    if selected_path is None:
        parser.error("Please specify the training progress CSV file with positional path or --progress-csv.")

    convert_training_progress(
        path=selected_path,
        target_column=args.target_column,
        job_dir=args.job_dir,
        scaler_path=args.scaler,
        output=args.output,
    )


if __name__ == "__main__":
    main()
