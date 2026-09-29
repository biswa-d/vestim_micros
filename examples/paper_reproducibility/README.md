# Portable paper inference

Copy this directory to another machine. No installation of `vestim`, GUI,
training code, or repository-root requirements is needed. Python 3.10+ and
these inference dependencies are required (initial installation needs internet).

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python run_offline_inference.py --job-dir job_20260122-104126_LG_NMC_best_FNN_with_Filteres --test-file LG_NMC_test_data/10_UDDS_40C.csv
```

Alternatively, run `run_test.bat` on Windows or `bash run_test.sh` on Linux/macOS.
These launchers create a local environment, install the inference requirements,
and forward arguments to the script. Paths in launcher arguments are relative
to this directory. When invoking Python directly, paths are relative to the
current working directory. CPU is the default; pass `--device cuda` to use CUDA.
The scaler dependency is pinned to the bundled scaler's saved version (1.7.2).
For a different job, use the scikit-learn version that produced its scaler.

## Required files to share

Each job must include:

```text
job_A/
  job_metadata.json
  augmentation_metadata.json       # required if training used augmentation
  scalers/augmentation_scaler.joblib # required if training used normalization
  models/FNN_90_45/repeat_3/
    task_info.json
    best_model.pth                  # or best_model_export.pt, a state-dict checkpoint
```

`task_info.json` supplies the architecture, features, target, and hyperparameters.
The script defines FNN, LSTM, and GRU locally and loads their weights. It uses
job-local files, not the training machine paths saved in metadata. A trained task
folder alone is insufficient: retain the parent job metadata and scaler.
Only load job artifacts from a source you trust; scalers use joblib serialization.

Supply the same **raw, unnormalized CSVs** used by the repository's standalone
workflow. Keep all scaler columns and filter source columns, including `Power`
for the bundled job. Test CSVs are not tracked by Git; distribute them separately
or include `LG_NMC_test_data` when sharing a folder archive. Model metadata is
explicitly allowed by `.gitignore`.

## Select a job or compare jobs

Test all CSVs in a directory:

```sh
python run_offline_inference.py --job-dir job_A --test-dir LG_NMC_test_data
```

Compare multiple trained jobs on identical data:

```sh
python run_offline_inference.py --job-dir job_A job_B --test-file LG_NMC_test_data/10_UDDS_40C.csv --output-dir comparison_output
```

You may also repeat `--job-dir`. Use distinct job folder names. With one job,
`--model-dir job_A/models/FNN_90_45/repeat_3` selects a specific trained task.
Multiple trained task folders require explicit selection; prepare one selected
trained task per job for multi-job comparisons. A zero-argument run works only
when exactly one `job_*` and one `*_test_data` directory sit beside the script.
Ambiguous choices fail instead of silently choosing a model or dataset.

## Outputs and plots

A single job defaults to `job_A/inference_output/`; multiple jobs default to
`comparison_output/`, with separate job subdirectories. `--output-dir` overrides
the root. Each job produces:

- Per-test `predictions.csv` and `prediction_plot.png` (target and error panels).
- `all_predictions.csv` and `summary.json`, including metrics and effective warmup.
- The output root contains `comparison_metrics.csv` for all requested jobs/tests.
- Multiple jobs additionally produce overlaid prediction/error plots per test and
  target. Axes use sample index; different target types are plotted separately.

Voltage RMSE/MAE and errors are in mV, SOC errors in percentage points; prediction
and target columns retain physical units. Error is measured minus predicted.
Pass `--skip-plot` for CSV/JSON only. Plots are saved, not opened interactively.
Reusing an output directory overwrites matching outputs; use a fresh directory
for each paper run. Aggregates include only the current run's successful inputs.

## Supported workflow and parity limits

The script reproduces causal Butterworth input filters from
`augmentation_metadata.json`, saved normalization, per-file recurrent state reset,
and saved post-inference moving average, exponential moving average, or
Savitzky-Golay filters. RNN warmup uses saved `LOOKBACK`, matching the repository's
standalone manager; `--warmup-samples` overrides it. FNN needs no warmup.

Supported architectures are the repository's fixed-width FNN, LSTM, and GRU
(state-dict checkpoints, including FNN activation/layer normalization and GRU
layer normalization). Missing metadata/scalers and unsupported model variants
fail explicitly. Full pickled model objects, LSTM_EMA/LSTM_LPF, variable-width RNNs,
resampling, padding, and calculated-column augmentation are not supported by this
runner. The bundled FNN job uses none of these unsupported options. Add and
validate support before sharing a comparative job that needs them.

Matching results requires identical weights, metadata, raw data, warmup, and
inference filter settings. GUI filter overrides are not stored back into every
job. Floating-point differences across CPU/GPU and library versions are possible.

## Bundled FNN versus LSTM example

The LSTM job `job_20260602-154104_best_LG_NMC_best_LSTM_without_Filteres`
uses Power, Battery_Temp_degC, and SOC, its own scaler, no input filters, and
400 warmup samples from its saved LOOKBACK. Both jobs can use the same raw CSVs.
From this directory:

```sh
python run_offline_inference.py --job-dir job_20260122-104126_LG_NMC_best_FNN_with_Filteres job_20260602-154104_best_LG_NMC_best_LSTM_without_Filteres --test-file LG_NMC_test_data/10_UDDS_40C.csv --output-dir comparison_output
```

Replace `--test-file LG_NMC_test_data/10_UDDS_40C.csv` with
`--test-dir LG_NMC_test_data` to process all cycles. To plot only the LSTM,
pass only its folder to `--job-dir`. Recurrent CPU inference is slower than FNN
inference because it advances the hidden state one sample at a time.

The LSTM scaler was saved with scikit-learn 1.2.2, whereas the FNN scaler uses
1.7.2. The combined run was validated with the installed 1.9.0 runtime, which
warns about both saved versions. The existing requirements pin matches the FNN
artifact; it is not a guarantee of cross-version compatibility for other jobs.

## ECM and a clean sharing bundle

The runner also accepts `--ecm-dir ecm_1rc_lg_nmc`, alone or together with ML
`--job-dir` arguments. See [ECM_README.md](ECM_README.md) for commands, parameter
provenance, optional paper windows/bias correction, and the unresolved difference
between the supplied timing benchmark and the historical paper ECM exports.
`python build_share_bundle.py` creates a separate minimal distribution without
removing the original source material.
