# Offline inference – paper reproducibility

This folder contains a self-contained inference script that can reproduce the paper results from a saved job folder without importing the repository code. It is intended for a new machine or a portable reproduction setup where you only have the trained job directory and test CSVs.

## What this script does

The script mirrors the repo's standalone testing flow:

- loads the model from the saved job folder
- reads `job_metadata.json` and `augmentation_metadata.json`
- applies the same filter augmentation used during training/testing
- normalizes all scaler columns with the saved `MinMaxScaler`
- runs inference on the normalized inputs
- denormalizes the target before computing RMSE, MAE, and R²

This is the exact logic that matches the repo's standalone results for the shipped job folders.

## System requirements

The script requires a working Python environment with the usual ML stack:

- Python 3.10+ recommended
- `torch`
- `pandas`
- `numpy`
- `scipy`
- `scikit-learn`
- `joblib`
- `matplotlib` (optional for plots)

If you are on Windows and the repo already has a local virtual environment, use it. If not, create a fresh one.

## Recommended setup on a new machine

### Option A: use the repo's existing environment

From the repo root:

```powershell
cd c:\Biswanath_Phd\vestim_micros
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\build_env\Scripts\Activate.ps1
python -V
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the repo environment exists and is already configured, this is the safest route.

### Option B: create a fresh virtual environment

```powershell
cd c:\Biswanath_Phd\vestim_micros
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Option C: minimal manual install

If you only need this offline script and do not want the full repo stack, install the required packages directly:

```powershell
python -m pip install torch pandas numpy scipy scikit-learn joblib matplotlib
```

## Common module errors and how to avoid them

### `ModuleNotFoundError: No module named 'torch'`

This usually means you are running the script with the system Python instead of the project virtual environment.

Fix:

```powershell
.\build_env\Scripts\Activate.ps1
# or
.\.venv\Scripts\Activate.ps1
python -c "import torch; print(torch.__version__)"
```

Then run the script again.

### `ModuleNotFoundError: No module named 'scipy'` or `sklearn`

Install the requirements in the active environment:

```powershell
python -m pip install -r requirements.txt
```

### `InconsistentVersionWarning` from scikit-learn

This warning is not usually fatal, but it means the scaler may have been saved with a different scikit-learn version than the one currently installed. It is harmless for inference in most cases, but the safest approach is to use the same environment that created the job.

## Portable folder layout

Use a folder like this:

```text
paper_run/
├── run_offline_inference.py
├── job_20260122-104126_LG_NMC_best_FNN_with_Filteres/
│   ├── job_metadata.json
│   ├── augmentation_metadata.json
│   ├── hyperparams.json
│   ├── scalers/
│   │   └── augmentation_scaler.joblib
│   └── models/
│       └── FNN_90_45/
│           └── B4096_Adam_LR_RLROP_VP180_rep_3/
│               ├── task_info.json
│               ├── best_model.pth
│               └── ...
├── test_data/
│   └── 10_UDDS_40C.csv
└── output/
```

The script can be run in either of these ways:

### One job + one test file

```powershell
python run_offline_inference.py --job-dir job_20260122-104126_LG_NMC_best_FNN_with_Filteres --test-file test_data\10_UDDS_40C.csv --skip-plot --device cpu
```

### One job + a whole test-data directory

```powershell
python run_offline_inference.py --job-dir job_20260122-104126_LG_NMC_best_FNN_with_Filteres --test-dir test_data --skip-plot --device cpu
```

### Zero-arg mode (if the script is beside a single `job_*` folder and a matching `*_test_data` folder)

```powershell
python run_offline_inference.py
```

## Important notes for reproducibility

- This script is self-contained and does not import the repo package.
- The job folder is the source of truth: model metadata, scaler, and feature columns all come from the job.
- The raw test CSV must contain the original columns used during augmentation, especially `Power`, because the script re-applies the filter augmentation recorded in `augmentation_metadata.json`.
- If a job folder contains multiple repeat folders, the script resolves the first valid trained model folder deterministically.
- If you only have the final best model task directory, it will still work as long as it contains `task_info.json` and `best_model.pth` or `best_model_export.pt`.

## Quickest validation run

From the folder containing the script and job folder:

```powershell
python run_offline_inference.py --job-dir job_20260122-104126_LG_NMC_best_FNN_with_Filteres --test-file .\test_data\10_UDDS_40C.csv --skip-plot --device cpu
```

This produces an `inference_output` folder with per-file metrics and prediction files.

## Output files

The script writes results into `inference_output/`:

- `summary.json`
- `all_predictions.csv`
- per-file folders with `predictions.csv`
- optional plot png files

That gives a clear paper-style reproducibility bundle without needing to run the repo GUI or training pipeline.

## Summary

If you want a clean reproduction on a new machine:

1. activate the correct environment
2. ensure dependencies are installed
3. copy the job folder and the raw test CSV into a portable folder
4. run `python run_offline_inference.py --job-dir ... --test-file ...`

This is the safest way to reproduce the paper results without depending on the rest of the codebase.
