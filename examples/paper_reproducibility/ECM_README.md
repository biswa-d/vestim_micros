# Portable LG NMC 1RC ECM

The Python ECM evaluates new test inputs using six fitted parameter tables.
No MATLAB installation, training code, or `ECM_LG_NMC_Test_Time` directory is
needed. Keep `ecm_inference.py` and `ecm_1rc_lg_nmc/` beside the main runner.

## Run ECM alone or alongside ML

From this directory:

```sh
python run_offline_inference.py --ecm-dir ecm_1rc_lg_nmc --test-file LG_NMC_test_data/10_UDDS_40C.csv
python run_offline_inference.py --job-dir job_20260122-104126_LG_NMC_best_FNN_with_Filteres job_20260602-154104_best_LG_NMC_best_LSTM_without_Filteres --ecm-dir ecm_1rc_lg_nmc --test-file LG_NMC_test_data/10_UDDS_40C.csv --output-dir comparison_output
```

Replace `--test-file ...` with `--test-dir LG_NMC_test_data` for all raw cycles.
Each model gets separate predictions/plots and the output root gets comparison
plots and metrics. Supply measured Current, SOC, Battery_Temp_degC, and Voltage;
ECM does not use the FNN filters or either ML scaler. Current is negative during
discharge. SOC may be a fraction or percent, using the MATLAB threshold of 1.5.

## Equations and preprocessing

The supplied timing benchmark is ported directly:

- Six temperature tables (-20, -10, 0, 10, 25, 40 C) contain SOC, OCV, R0, R1, C1.
- Tables are interpolated/extrapolated onto the first (-20 C) SOC grid.
- R0, R1, and OCV use bilinear interpolation in SOC/temperature; C1 uses bilinear
  interpolation of log(C1), then exponentiation. Queries are clipped to the grids.
- RC voltage starts at zero for each file. For each subsequent sample,
  `a = exp(-dt / (R1_previous * C1_previous))` and
  `Vrc = a * Vrc_previous + R1_previous * (1-a) * Current_previous`.
- Terminal voltage is `OCV + R0 * Current + Vrc`.

For original MATLAB CSVs with `Adjusted_Test_Time(s)` (or its MATLAB-safe alias),
invalid rows and duplicate timestamps are removed, and all inputs are linearly
resampled at one second, as in the benchmark. Those CSVs may use `Voltage(V)`
and `Current(A)` column names. For the shared ML CSVs, each row advances the ECM
by the configured `sample_interval_s=1`; their raw `Time` column can reset and
is deliberately not used. The effective time policy is saved with the metrics.
Output tables use the processed frame so resampled predictions remain aligned.

## Historical paper outputs are not yet reproduced exactly

The port matches all **30 saved timing-benchmark cases**, including row counts
and RMSE (maximum observed discrepancy below 1e-7 mV). However,
`test_results_all_files_1RC` was produced using different, currently unconfirmed
settings. It contains full-cycle results with essentially zero mean error; the
provided timing code uses discharge-only inputs and a -20 C master SOC grid.
Its default predictions therefore must not be presented as an exact recreation
of those historical paper exports. The original generating script is needed to
resolve their SOC grid and preprocessing/calibration choices.

`--ecm-bias-correction` explicitly subtracts the full test file's mean prediction
error using measured Voltage. It is an evaluation-time calibration that uses
labels, not an independent deployable prediction. Raw predictions remain in
`Raw_Predicted_Voltage`; raw full-file RMSE/bias and the correction flag are saved.
This option alone does not resolve the historical discrepancy.

`--paper-window` applies the drive-cycle/temperature endpoint table from
`E66_voltage_plot_ECM.m` to **every** selected model's metrics and plots. Full
inference and any requested bias correction run before this common truncation.
Without this option, metrics cover the full file. Plot errors follow the runner's
measured-minus-predicted convention; the historical MATLAB exports use the
opposite sign. RMSE and MAE are unaffected by that sign convention.

## Sharing a clean directory

From the development copy, build a fresh bundle:

```sh
python build_share_bundle.py
```

The generated `share_bundle/` includes only runtime scripts/docs, two selected
ML job artifacts, ECM configuration and six MAT parameter tables, raw test CSVs,
and a checksum manifest. It excludes MATLAB scripts, discharge-only benchmark
inputs, timing outputs, archived predictions, training logs, and redundant
checkpoints. The source directories are preserved. Copy or zip `share_bundle/`
for recipients; they install `requirements.txt` and run the same commands above.
The builder refuses to overwrite an existing bundle; choose a fresh
`--output-dir` when rebuilding.
