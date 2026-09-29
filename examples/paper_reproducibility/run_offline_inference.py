#!/usr/bin/env python3
"""
Standalone offline inference for a trained VEstim model (FNN / LSTM / GRU).

Place a job_* folder and a *_test_data folder alongside this script and run:
    python run_offline_inference.py

Or with explicit paths:
    python run_offline_inference.py --job-dir job_xxx --test-dir my_test_data

NO dependency on the vestim package.  All model definitions and inference
logic are self-contained in this single file.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ------------------------------------------------------------------  matplotlib is optional
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

# ==================================================================
#  Model definitions  (copied from vestim â€“ self-contained)
# ==================================================================

class FNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_sizes, dropout_prob=0.0,
                 apply_clipped_relu=False, activation_function='ReLU', use_layer_norm=False):
        super().__init__()
        layers = []
        cur = input_size
        act = nn.GELU() if activation_function == 'GELU' else nn.ReLU()
        for hs in hidden_layer_sizes:
            layers.append(nn.Linear(cur, hs))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hs))
            layers.append(act)
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))
            cur = hs
        layers.append(nn.Linear(cur, output_size))
        if apply_clipped_relu:
            layers.extend([nn.ReLU(inplace=True), nn.Hardtanh(0, 1)])
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

    def forward(self, x):
        return self.network(x)  # [B, in] -> [B, out]


class _GRULN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, hx=None):
        x, hx = self.gru(x, hx)
        return self.layer_norm(x), hx


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_units, num_layers, output_size=1,
                 dropout_prob=0.0, device='cpu', apply_clipped_relu=False, use_layer_norm=False):
        super().__init__()
        self.device = device
        self.apply_output_dropout = (num_layers > 1 and dropout_prob > 0)
        if use_layer_norm:
            self.gru = _GRULN(input_size, hidden_units, num_layers,
                              dropout_prob if num_layers > 1 else 0).to(device)
        else:
            self.gru = nn.GRU(input_size, hidden_units, num_layers,
                              batch_first=True,
                              dropout=dropout_prob if num_layers > 1 else 0).to(device)
        self.dropout = nn.Dropout(dropout_prob).to(device) if dropout_prob > 0 else None
        self.fc = nn.Linear(hidden_units, output_size).to(device)
        self.final_activation = nn.Hardtanh(0, 1) if apply_clipped_relu else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(p.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(p.data)
            elif 'bias' in name:
                nn.init.constant_(p.data, 0.0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x, h_0=None):
        x = x.to(self.device)
        out, h_n = self.gru(x, h_0) if h_0 is not None else self.gru(x)
        if self.apply_output_dropout and self.dropout is not None:
            out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return self.final_activation(out), h_n


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_units, num_layers, device,
                 dropout_prob=0.0, apply_clipped_relu=False, use_layer_norm=False):
        super().__init__()
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_units, num_layers,
                            batch_first=True,
                            dropout=dropout_prob if num_layers > 1 else 0).to(device)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_units, 1).to(device)
        self.apply_output_dropout = (num_layers > 1)
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(p.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(p.data)
            elif 'bias' in name:
                nn.init.constant_(p.data, 0.0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.01)

    def forward(self, x, h_s=None, h_c=None):
        x = x.to(self.device)
        if h_s is None or h_c is None:
            out, (h_s, h_c) = self.lstm(x)
        else:
            out, (h_s, h_c) = self.lstm(x, (h_s, h_c))
        if self.apply_output_dropout:
            out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out, (h_s, h_c)


# ==================================================================
#  Helpers
# ==================================================================

def _load_json(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_device(dev: str) -> torch.device:
    if dev.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def _error_multiplier(y_true: np.ndarray, target: str) -> float:
    t = target.lower()
    if "voltage" in t:
        return 1000.0
    if "soc" in t:
        return 100.0 if np.max(np.abs(y_true)) <= 5.0 else 1.0
    return 1.0


def _find_job_dir(provided: str | None) -> Path:
    if provided:
        p = Path(provided).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Job directory not found: {p}")
        return p
    here = Path(__file__).resolve().parent
    candidates = sorted(p for p in here.glob("job_*") if p.is_dir())
    if not candidates:
        raise FileNotFoundError("No job_* folder found. Pass --job-dir.")
    if len(candidates) > 1:
        raise ValueError("Multiple jobs found. Select with --job-dir.")
    return candidates[0]


def _collect_test_files(test_path: str | None) -> list[Path]:
    if test_path is None:
        here = Path(__file__).resolve().parent
        candidates = [d for d in sorted(here.iterdir())
                      if d.is_dir() and d.name.lower().endswith("_test_data")]
        if len(candidates) > 1:
            raise ValueError("Multiple test folders found. Pass --test-dir or --test-file.")
        if candidates:
            test_path = str(candidates[0])
        if test_path is None:
            raise FileNotFoundError("No *_test_data folder found. Pass --test-dir or --test-file.")
    p = Path(test_path).expanduser().resolve()
    if p.is_file():
        return [p]
    if p.is_dir():
        csvs = sorted(p.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV files in {p}")
        return csvs
    raise FileNotFoundError(f"Test path not found: {p}")


def _find_model_dir(job_dir: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Model directory not found: {p}")
        return p
    models_root = job_dir / "models"
    if not models_root.exists():
        raise FileNotFoundError(f"No models/ in {job_dir}")
    candidates: list[Path] = []
    for tj in sorted(models_root.rglob("task_info.json")):
        parent = tj.parent
        if (parent / "best_model.pth").exists() or (parent / "best_model_export.pt").exists():
            candidates.append(parent)
    if not candidates:
        raise FileNotFoundError(
            f"No trained model found under {models_root}. "
            "Expected a task folder containing task_info.json and "
            "best_model.pth / best_model_export.pt."
        )
    if len(candidates) > 1:
        raise ValueError("Multiple models found. Select with --model-dir: " +
                         ", ".join(str(c) for c in candidates))
    return candidates[0]


def _build_task(job_dir: Path, model_dir: Path) -> dict[str, Any]:
    ti = _load_json(model_dir / "task_info.json")
    jm = _load_json(job_dir / "job_metadata.json")
    hp = dict(ti.get("hyperparams", {}))
    mm = dict(ti.get("model_metadata", {}))
    dl = dict(ti.get("data_loader_params", {}))

    mtype = str(mm.get("model_type") or hp.get("MODEL_TYPE") or "").upper()
    if mtype not in {"FNN", "LSTM", "GRU"}:
        raise ValueError(f"Unsupported model type: {mtype}. Supported: FNN, LSTM, GRU.")

    best = model_dir / "best_model.pth"
    if not best.exists():
        best = model_dir / "best_model_export.pt"
    if not best.exists():
        raise FileNotFoundError(f"No model file in {model_dir}")

    feats = dl.get("feature_columns") or hp.get("FEATURE_COLUMNS")
    targ = dl.get("target_column") or hp.get("TARGET_COLUMN")
    if not feats or not targ:
        raise ValueError("Missing feature/target columns in task_info.json")

    return {
        "task_name": model_dir.name,
        "model_type": mtype,
        "best_model_path": str(best),
        "task_dir": str(model_dir),
        "hyperparams": hp,
        "job_folder_augmented_from": str(job_dir),
        "model_metadata": {"model_type": mtype, **mm},
        "data_loader_params": {"feature_columns": feats, "target_column": targ, **dl},
        "job_metadata": jm,
    }


def _load_model(task: dict, device: torch.device) -> nn.Module:
    hp = task["hyperparams"]
    mtype = task["model_type"]
    model_path = task["best_model_path"]

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    input_size = hp["INPUT_SIZE"] if "INPUT_SIZE" in hp else len(task["data_loader_params"]["feature_columns"])

    if mtype == "FNN":
        hidden_sizes = hp.get("HIDDEN_LAYER_SIZES", [64, 32])
        if isinstance(hidden_sizes, str):
            hidden_sizes = [int(x.strip()) for x in hidden_sizes.split(",")]
        dropout = hp.get("DROPOUT_PROB", 0.0)
        if isinstance(dropout, str):
            dropout = float(dropout)
        # The repo enables a clipped ReLU output (ReLU + Hardtanh(0,1)) whenever
        # normalization was applied, since the target is in [0, 1].
        apply_clipped = bool(
            hp.get("normalization_applied", False)
            or task["job_metadata"].get("normalization_applied", False)
        )
        model = FNNModel(input_size, 1, hidden_sizes, dropout_prob=dropout,
                         apply_clipped_relu=apply_clipped,
                         activation_function=hp.get("FNN_ACTIVATION", "ReLU"),
                         use_layer_norm=bool(hp.get("FNN_USE_LAYERNORM", False)))
    else:
        hidden = int(hp.get("HIDDEN_UNITS", 64))
        layers = int(hp.get("LAYERS", 2))
        dropout = hp.get("DROPOUT_PROB", hp.get("LSTM_DROPOUT_PROB", 0.0))
        if isinstance(dropout, str):
            dropout = float(dropout)
        if mtype == "GRU":
            model = GRUModel(input_size, hidden, layers, output_size=1,
                             dropout_prob=dropout, device=device,
                             apply_clipped_relu=bool(hp.get("normalization_applied", False)),
                             use_layer_norm=bool(hp.get("GRU_USE_LAYERNORM", False)))
        else:
            model = LSTMModel(input_size, hidden, layers, device, dropout_prob=dropout)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif hasattr(checkpoint, "state_dict"):
        model = checkpoint
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    print(f"  Model loaded: {mtype}  params={sum(p.numel() for p in model.parameters()):,}")
    return model


def _load_scaler(task: dict) -> Any | None:
    jm = task.get("job_metadata", {})
    if not jm.get("normalization_applied"):
        return None
    rel = jm.get("scaler_path", "scalers/augmentation_scaler.joblib")
    job_dir = Path(task["job_folder_augmented_from"])
    rel = str(rel).replace(chr(92), "/")
    relative = Path(rel)
    if relative.is_absolute() or ":" in rel or ".." in relative.parts:
        relative = Path("scalers") / relative.name
    sp = job_dir / relative
    if not sp.exists():
        sp = job_dir / "scalers" / Path(rel).name
    if not sp.exists():
        raise FileNotFoundError(f"Training used normalization; required scaler missing: {sp}")
    sc = joblib.load(sp)
    print(f"  Scaler loaded: {sp.name}")
    return sc


def _inverse_y(values: np.ndarray, scaler, target_col: str, fallback=None) -> np.ndarray:
    if scaler is None:
        return values
    try:
        fn = getattr(scaler, "feature_names_in_", None)
        cols = list(fn) if fn is not None else list(fallback or [])
    except Exception:
        cols = []
    if target_col not in cols:
        return values
    idx = cols.index(target_col)
    if hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
        data_min = np.asarray(scaler.data_min_).flatten()
        data_max = np.asarray(scaler.data_max_).flatten()
        lo, hi = data_min[idx], data_max[idx]
        if tuple(scaler.feature_range) != (0, 1):
            raise ValueError("Repository parity requires MinMaxScaler feature_range=(0, 1)")
        return values * (hi - lo) + lo
    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        return values * scaler.scale_[idx] + scaler.mean_[idx]
    raise ValueError(f"Unsupported scaler: {type(scaler).__name__}")


def _scaler_columns(scaler, fallback: list[str] | None) -> list[str]:
    """Return the columns (and their order) the scaler was fitted on."""
    try:
        fn = getattr(scaler, "feature_names_in_", None)
        if fn is not None and len(fn) > 0:
            return list(fn)
    except Exception:
        pass
    return list(fallback or [])


def _apply_augmentation(df: pd.DataFrame, job_dir: Path) -> pd.DataFrame:
    """Re-apply the augmentation steps recorded in augmentation_metadata.json.

    Mirrors StandaloneTestingManager._apply_automatic_augmentation.  The step
    implemented offline is the causal Butterworth low-pass filter (lfilter),
    which the paper job uses to derive P_filter_2 / P_filter_0p2 from Power.
    """
    meta_path = job_dir / "augmentation_metadata.json"
    if not meta_path.exists():
        return df

    metadata = _load_json(meta_path)
    applied_filters = metadata.get("applied_filters", [])

    padding_info = metadata.get("padding", {})
    resampling_info = metadata.get("resampling", {})
    created_columns = metadata.get("created_columns", [])

    if padding_info.get("applied", False):
        raise NotImplementedError(
            "Offline inference does not support padding augmentation "
            f"({padding_info}). Run the repo augmentation pipeline to prepare the test file."
        )
    if resampling_info.get("applied", False):
        raise NotImplementedError(
            "Offline inference does not support resampling augmentation "
            f"({resampling_info}). Run the repo augmentation pipeline to prepare the test file."
        )
    if created_columns:
        raise NotImplementedError(
            "Offline inference does not support calculated columns "
            f"({[c.get('column_name') for c in created_columns]}). "
            "Run the repo augmentation pipeline to prepare the test file."
        )

    try:
        from scipy.signal import butter, lfilter
    except ImportError as e:
        raise ImportError(
            "scipy is required to reproduce the filter augmentation. "
            "Install it with: pip install scipy"
        ) from e

    result_df = df.copy()
    for cfg in applied_filters:
        col = cfg["column"]
        if col not in result_df.columns:
            available = ", ".join(result_df.columns.tolist())
            raise ValueError(
                f"Filter requires column '{col}' which is not in the test data. "
                f"Available columns: {available}"
            )
        corner = float(cfg["corner_frequency"])
        fs = float(cfg["sampling_rate"])
        order = int(cfg.get("filter_order", 4))
        out_col = cfg.get("output_column_name") or col

        nyquist = 0.5 * fs
        if corner >= nyquist:
            raise ValueError(
                f"Corner frequency ({corner} Hz) must be less than the "
                f"Nyquist frequency ({nyquist} Hz)."
            )
        b, a = butter(order, corner / nyquist, btype="low", analog=False)
        result_df[out_col] = lfilter(b, a, result_df[col].to_numpy(dtype=np.float64))

    return result_df


def _run_inference(model: nn.Module, model_type: str, scaler, task: dict,
                   test_file: Path, warmup: int, device: torch.device) -> dict | None:
    dl = task["data_loader_params"]
    feats = dl["feature_columns"]
    target = dl["target_column"]

    df = pd.read_csv(test_file)
    df = _apply_augmentation(df, Path(task["job_folder_augmented_from"]))

    for c in feats + [target]:
        if c not in df.columns:
            raise ValueError(f"Required column {c!r} not in {test_file}")

    # Mirror the repo pipeline: normalize every scaler-known column first,
    # run inference on the normalized features, then denormalize the target.
    normalization_applied = bool(task["job_metadata"].get("normalization_applied")) and scaler is not None
    if task["job_metadata"].get("normalization_applied") and scaler is None:
        print("  WARNING: training used normalization but the scaler is missing; results may be wrong")
    if normalization_applied:
        scaler_cols = _scaler_columns(scaler, task["job_metadata"].get("normalized_columns", []))
        if not scaler_cols:
            raise ValueError("Scaler column names missing from scaler and job metadata")
        else:
            missing_scaler = [c for c in scaler_cols if c not in df.columns]
            if missing_scaler:
                raise ValueError(f"Scaler-required columns missing: {missing_scaler}")
            df[scaler_cols] = scaler.transform(df[scaler_cols])

    df_num = df[list(dict.fromkeys(feats + [target]))].apply(pd.to_numeric, errors="raise")
    if df_num.empty or not np.isfinite(df_num.to_numpy()).all():
        raise ValueError(f"Empty data or non-finite inputs/target in {test_file}")
    orig_len = len(df_num)

    if model_type != "FNN" and warmup > 0:
        first_row = df_num.iloc[0:1]
        warmup_df = pd.concat([first_row] * warmup, ignore_index=True)
        df_full = pd.concat([warmup_df, df_num], ignore_index=True)
    else:
        warmup = 0
        df_full = df_num

    X = torch.tensor(df_full[feats].values.astype(np.float32),
                     device=device).view(-1, 1, len(feats))

    preds = []
    h_s, h_c = None, None
    with torch.no_grad():
        for t in range(len(df_full)):
            xt = X[t].unsqueeze(0)
            if model_type == "FNN":
                y = model(xt.view(1, -1))
            elif model_type == "GRU":
                y, h_s = model(xt, h_s)
            else:
                y, (h_s, h_c) = model(xt, h_s, h_c)
            if t >= warmup:
                preds.append(y.item())

    y_pred = np.array(preds, dtype=np.float32)[:orig_len]
    y_true = df_num[target].values[:orig_len].astype(np.float32)

    y_pred = _filter_predictions(y_pred, task["hyperparams"])
    if normalization_applied:
        y_pred = _inverse_y(y_pred, scaler, target, scaler_cols)
        y_true = _inverse_y(y_true, scaler, target, scaler_cols)

    mult = _error_multiplier(y_true, target)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)) * mult)
    mae = float(mean_absolute_error(y_true, y_pred) * mult)
    r2 = float(r2_score(y_true, y_pred))

    return {"predictions": y_pred, "true_values": y_true, "rmse": rmse, "mae": mae, "r2": r2}


def _save_outputs(out_dir: Path, raw_df: pd.DataFrame,
                  tv: np.ndarray, pv: np.ndarray,
                  target_col: str, model_type: str, skip_plot: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not len(raw_df) == len(tv) == len(pv):
        raise ValueError("Output frame and predictions must have matching lengths")
    n = len(tv)
    mult = _error_multiplier(tv[:n], target_col)

    df = raw_df.iloc[:n].copy()
    df[f"True_{target_col}"] = tv[:n]
    df[f"Predicted_{target_col}"] = pv[:n]
    df[f"Error_{target_col}"] = (tv[:n] - pv[:n]) * mult
    df["Model_Type"] = model_type
    df.to_csv(out_dir / "predictions.csv", index=False)

    if _HAS_MPL and not skip_plot:
        x = np.arange(n)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        ax1.plot(x, tv[:n], label="True", linewidth=1.8)
        ax1.plot(x, pv[:n], label="Predicted", linewidth=1.2)
        ax1.set_ylabel(target_col)
        ax1.set_title("Prediction vs. target")
        ax1.legend()
        ax2.plot(x, (tv[:n] - pv[:n]) * mult, color="tab:red", linewidth=1.2)
        ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax2.set_xlabel("Sample index")
        ax2.set_ylabel(f"Error ({_error_unit(target_col)})")
        ax2.set_title("Prediction error")
        fig.tight_layout()
        fig.savefig(out_dir / "prediction_plot.png", dpi=200)
        plt.close(fig)


# ==================================================================
#  Main
# ==================================================================

def _filter_predictions(values, hp):
    kind = hp.get("INFERENCE_FILTER_TYPE", "None")
    if kind in (None, "None", ""):
        return values
    if kind == "Moving Average":
        return pd.Series(values).rolling(int(hp["INFERENCE_FILTER_WINDOW_SIZE"]), min_periods=1).mean().values
    if kind == "Exponential Moving Average":
        return pd.Series(values).ewm(alpha=float(hp["INFERENCE_FILTER_ALPHA"]), adjust=False).mean().values
    if kind == "Savitzky-Golay":
        from scipy.signal import savgol_filter
        window = int(hp["INFERENCE_FILTER_WINDOW_SIZE"])
        order = int(hp["INFERENCE_FILTER_POLYORDER"])
        window += (window % 2 == 0)
        if window <= order:
            window = order + 1
            window += (window % 2 == 0)
        return savgol_filter(values, window, order)
    raise ValueError(f"Unsupported inference filter: {kind}")


def _warmup_samples(task, override):
    value = override if override is not None else task["hyperparams"].get(
        "LOOKBACK", task["data_loader_params"].get("lookback", 0))
    value = 0 if value in (None, "N/A") else int(value)
    if value < 0:
        raise ValueError("Warmup samples must be nonnegative")
    return 0 if task["model_type"] == "FNN" else value


def _error_unit(target):
    if "voltage" in target.lower():
        return "mV"
    if "soc" in target.lower():
        return "percentage points"
    return target


def _save_comparison(root, results):
    for test_name, entries in results.items():
        targets = {entry[1] for entry in entries}
        for target in sorted(targets):
            group = [entry for entry in entries if entry[1] == target]
            fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
            for label, _, result in group:
                actual, predicted = result["true_values"], result["predictions"]
                x = np.arange(len(actual))
                # Keep each job's actual curve: preprocessing may differ between jobs.
                top.plot(x, actual, linestyle="--", alpha=0.5, label=f"{label}: measured")
                top.plot(x, predicted, label=f"{label}: predicted")
                bottom.plot(x, (actual - predicted) * _error_multiplier(actual, target), label=label)
            top.set_ylabel(target)
            top.set_title(test_name)
            top.legend(fontsize=7)
            bottom.set_ylabel(f"Error ({_error_unit(target)})")
            bottom.set_xlabel("Sample index")
            bottom.axhline(0, color="black", linewidth=0.5)
            fig.tight_layout()
            safe_target = "".join(c if c.isalnum() else "_" for c in target)
            fig.savefig(root / f"comparison_{Path(test_name).stem}_{safe_target}.png", dpi=200)
            plt.close(fig)


def _paper_length(filename, available):
    """Exact endpoints from E66_voltage_plot_ECM.m, shared by all model types."""
    import re
    match = re.match(r"(?:\d+_)?(HWFET|LA92|UDDS|US06)_(n?\d+)C", filename)
    endpoints = {
        -20: (24677, 44378, 68848, 21982), -10: (27258, 48949, 77143, 22396),
        0: (19770, 54806, 46000, 19585), 10: (26995, 55484, 82120, 22581),
        25: (26728, 53065, 83226, 21532), 40: (27995, 55115, 84194, 23537),
    }
    if not match:
        raise ValueError(f"No LG NMC paper endpoint for {filename}")
    cycle, token = match.groups()
    temperature = -int(token[1:]) if token.startswith("n") else int(token)
    if temperature not in endpoints:
        raise ValueError(f"No paper endpoint for temperature {temperature}")
    return min(available, endpoints[temperature][("HWFET", "LA92", "UDDS", "US06").index(cycle)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Portable inference and comparison for saved PyBattML jobs")
    parser.add_argument("--job-dir", nargs="+", action="extend", help="One or more job folders; may be repeated")
    tests = parser.add_mutually_exclusive_group()
    tests.add_argument("--test-dir")
    tests.add_argument("--test-file")
    parser.add_argument("--ecm-dir", nargs="+", action="extend", help="Portable ECM model folder(s)")
    parser.add_argument("--ecm-bias-correction", action="store_true",
                        help="Subtract each ECM file's measured mean error (requires ground truth)")
    parser.add_argument("--paper-window", action="store_true",
                        help="Use the LG NMC paper's drive-cycle endpoints for every model")
    parser.add_argument("--model-dir", help="Select a task folder when testing a single job")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=1, help="CPU threads; 1 avoids overhead for sample-wise RNN inference")
    parser.add_argument("--warmup-samples", type=int, help="Override saved LOOKBACK (RNN only)")
    parser.add_argument("--skip-plot", action="store_true")
    args = parser.parse_args()
    if not args.skip_plot and not _HAS_MPL:
        parser.error("Plots require matplotlib. Install requirements.txt or pass --skip-plot.")
    jobs = [_find_job_dir(p) for p in args.job_dir] if args.job_dir else (
        [] if args.ecm_dir else [_find_job_dir(None)])
    ecm_dirs = [Path(p).expanduser().resolve() for p in args.ecm_dir or []]
    all_dirs = jobs + ecm_dirs
    if len({p.name for p in all_dirs}) != len(all_dirs):
        parser.error("All ML and ECM model folder names must be distinct")
    if args.ecm_bias_correction and not ecm_dirs:
        parser.error("--ecm-bias-correction requires --ecm-dir")
    if len(set(jobs)) != len(jobs) or len({p.name for p in jobs}) != len(jobs):
        parser.error("Job folders must have distinct names and paths")
    if args.model_dir and len(jobs) != 1:
        parser.error("--model-dir is only supported with a single --job-dir")
    test_files = _collect_test_files(args.test_file or args.test_dir)
    device = _resolve_device(args.device)
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)
    multiple = len(all_dirs) > 1
    root = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        Path(__file__).resolve().parent / "comparison_output" if multiple else all_dirs[0] / "inference_output")
    root.mkdir(parents=True, exist_ok=True)
    comparison, metrics = {}, []
    for job in all_dirs:
        is_ecm = job in ecm_dirs
        if is_ecm:
            # Explicit file loading also works in isolated Python (-I).
            import importlib.util
            spec = importlib.util.spec_from_file_location("portable_ecm", Path(__file__).with_name("ecm_inference.py"))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            model = module.ECMModel(job)
            model_dir, warmup, target, model_type = job, 0, "Voltage", "ECM_1RC"
        else:
            model_dir = _find_model_dir(job, args.model_dir)
            task = _build_task(job, model_dir)
            model, scaler = _load_model(task, device), _load_scaler(task)
            warmup = _warmup_samples(task, args.warmup_samples)
            target, model_type = task["data_loader_params"]["target_column"], task["model_type"]
        out = root / job.name if multiple else root
        out.mkdir(parents=True, exist_ok=True)
        print(f"Job: {job.name}; device: {device}; warmup: {warmup}")
        per_file, parts = [], []
        for tf in test_files:
            result = model.run(tf, args.ecm_bias_correction) if is_ecm else _run_inference(
                model, model_type, scaler, task, tf, warmup, device)
            frame = result["frame"].copy() if is_ecm else pd.read_csv(tf)
            if args.paper_window:
                length = _paper_length(tf.name, len(frame))
                frame = frame.iloc[:length].copy()
                for key in ("predictions", "true_values", "raw_predictions", "time_s"):
                    if key in result:
                        result[key] = result[key][:length]
                actual, predicted = result["true_values"], result["predictions"]
                multiplier = _error_multiplier(actual, target)
                result.update(rmse=float(np.sqrt(mean_squared_error(actual, predicted))*multiplier),
                              mae=float(mean_absolute_error(actual, predicted)*multiplier),
                              r2=float(r2_score(actual, predicted)))
            if is_ecm:
                frame["Raw_Predicted_Voltage"] = result["raw_predictions"]
            _save_outputs(out / tf.stem, frame, result["true_values"],
                          result["predictions"], target, model_type, args.skip_plot)
            row = dict(job=job.name, test_file=tf.name, model_type=model_type,
                       target=target, error_unit=_error_unit(target), samples=len(result["predictions"]),
                       **{k: result[k] for k in ("rmse", "mae", "r2")})
            row["evaluation_window"] = "paper" if args.paper_window else "full_file"
            if is_ecm:
                row.update(time_policy=result["time_policy"],
                           bias_correction_applied=result["bias_correction_applied"],
                           full_file_raw_bias_mv=result["raw_bias_mv"],
                           full_file_raw_rmse_mv=result["raw_rmse_mv"])
            per_file.append(row)
            metrics.append(row)
            print(f"  {tf.name}: RMSE={row['rmse']:.6f} {row['error_unit']}, R2={row['r2']:.6f}")
            frame = pd.read_csv(out / tf.stem / "predictions.csv")
            frame.insert(0, "Source_File", tf.name)
            parts.append(frame)
            if multiple and not args.skip_plot:
                comparison.setdefault(tf.name, []).append((job.name, target, result))
        pd.concat(parts, ignore_index=True).to_csv(out / "all_predictions.csv", index=False)
        summary = dict(job_dir=str(job), model_dir=str(model_dir), model_type=model_type,
                       output_dir=str(out), device=str(device), warmup_samples=warmup,
                       timestamp=datetime.now().isoformat(), per_file=per_file)
        summary["evaluation_window"] = "paper" if args.paper_window else "full_file"
        if is_ecm:
            summary["ecm_config"] = model.config
            summary["bias_correction_applied"] = args.ecm_bias_correction
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(metrics).to_csv(root / "comparison_metrics.csv", index=False)
    if comparison:
        _save_comparison(root, comparison)
    print(f"Results: {root}")


if __name__ == "__main__":
    main()
