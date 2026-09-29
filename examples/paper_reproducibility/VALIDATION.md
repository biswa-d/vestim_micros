# Offline inference validation

Validated on Windows, CPU, Python 3.12, on 2026-09-21.

- Bundled FNN job: all 24 raw LG NMC drive cycles completed (1,645,214 samples).
- All 24 prediction plots and per-file CSVs were generated.
- Every prediction was compared with the saved repository outputs in
  `new_test_result_20260122_134044`; maximum absolute difference was
  0.000501001 mV (approximately 5.01e-7 V).
- `10_UDDS_40C.csv`: RMSE 12.450037 mV.
- A separate 2,000-sample test compared the offline runner directly with the
  current repository continuous inference service. It passed at 1e-6 V absolute
  prediction tolerance and four decimal places for RMSE in mV.
- Five focused regression tests passed: reference-service parity, model/state
  parity for FNN/LSTM/GRU (including layer-normalized GRU), all saved prediction
  filter types, configuration failures/warmup, and portable multi-job CLI use.
- The multi-job test used two synthetic jobs in a temporary directory, isolated
  Python (`-I`), and only a copied runner plus job artifacts/data. It checked
  separate outputs, comparison metrics/plots, and failure for a missing scaler.
  Real FNN/LSTM comparison was subsequently validated as recorded below.

Run the developer regression tests from the repository root:

```sh
python -m unittest discover -s test -p test_offline_inference.py -v
```

The runner has no `vestim` imports. The developer tests intentionally import the
repository reference implementations; they are not needed by recipients.

Validation runtime: torch 2.10.0+cu128 (CPU execution), numpy 2.1.2, pandas 3.0.3,
scipy 1.18.0, scikit-learn 1.9.0, joblib 1.5.3, matplotlib 3.11.0. The saved scaler
was produced with scikit-learn 1.7.2; this validation emitted a version warning.
The portable requirements pin 1.7.2 to match the saved artifact. A fresh dependency
installation and Linux/macOS launchers were not executed in this validation.

Local full-run outputs: `output/offline_validation/all_cycles/` at the repository
root; per-cycle reference differences: `output/offline_validation/reference_comparison.csv`.
Generated outputs and test data remain excluded from Git.

## Real LSTM job follow-up

Ran both bundled jobs on the complete `10_UDDS_40C.csv` (103,775 samples), CPU,
with separate per-job plots and an overlaid comparison plot:

| Job | Warmup samples | RMSE (mV) | R2 |
| --- | ---: | ---: | ---: |
| FNN with filters | 0 | 12.450037 | 0.998394 |
| LSTM without filters | 400 | 18.071740 | 0.996616 |

The LSTM predictions differed from both saved repository UDDS exports
(`new_test_result_20260615_064918` and `new_test_result_20260824_142231`) by at
most 0.021794 mV. This is an observed tolerance, not bitwise identity; the source
of that small difference was not isolated. The LSTM scaler was saved with
scikit-learn 1.2.2 and emitted a version warning in the installed 1.9.0 runtime.
All-cycle LSTM evaluation was not run. Results are under this example's
`comparison_output/` directory. No runner change was needed to load this job.

## ECM integration (2026-09-22)

- Ported the provided 1RC benchmark equations and six Highest parameter tables
  into a Python-only runtime. No MATLAB or repository package is required.
- All 30 rows in the saved MATLAB `PerFile_Inference_Timing.csv` matched sample
  counts, RMSE, and bias to seven decimal places in mV; the maximum observed
  RMSE difference was approximately 7.11e-14 mV.
- Four ECM tests passed (analytical RC step response/state reset, all benchmark
  cases, adjusted-time alignment, and isolated CLI/bias-correction execution).
  The five existing offline inference tests also passed.
- The complete UDDS 40 C cycle was evaluated with FNN, LSTM, and ECM together;
  individual plots, a combined plot, and comparison metrics were generated in
  `comparison_output_ecm/`. Full-file RMSEs were 12.450037, 18.071740, and
  73.407301 mV respectively. ECM here is the uncorrected benchmark formulation,
  not a verified reproduction of the historical paper ECM curve.
- A clean 48-file sharing bundle was built (176,568,245 bytes before its
  checksum manifest), excluding the original MATLAB/benchmark/results directory.
  Its ECM runner completed the same full UDDS cycle using isolated Python (-I),
  with RMSE 73.407301 mV. The bundle contains both selected ML models, six ECM
  parameter tables, and all 24 shared raw test CSVs.
- CPU thread count now defaults to one for sample-wise RNN inference to avoid
  thread-pool overhead; `--cpu-threads` can override it. This full-cycle run kept
  the previously measured FNN/LSTM RMSEs to six decimal places.

**Unresolved historical parity:** the old `test_results_all_files_1RC` exports
have essentially zero full-file mean error and a different low-SOC response.
The supplied timing script uses the -20 C master SOC grid (minimum 16% SOC)
with query clipping, unlike the response observed in those older exports.
Changing SOC grids and enabling bias correction can change the result, but no
unconfirmed configuration has been labeled as an exact historical reproduction.
The original generating script is needed before claiming paper-result parity.
