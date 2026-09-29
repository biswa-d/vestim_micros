# PyBattML Paper Research and Positioning Brief

**Status:** Working research brief for an applied battery-software paper  
**Research snapshot:** 28 September 2026  
**Purpose:** Position PyBattML as a practical environment for battery researchers to explore, compare, preserve, and integrate machine-learning models.

## Core framing

You are right to frame the tool more broadly than standalone inference. The central utility is a **research workflow**: start from battery measurements, prepare features and targets, explore model/hyperparameter choices and repetitions, select a model based on validation, preserve the complete experiment in a job directory, and carry the chosen model into a battery-system simulation or analysis workflow.

A job directory is valuable because it can bind together more than weights: task/model hierarchy, feature order, target name, hyperparameters, training history, checkpoints, normalization metadata, scaler, augmentation configuration, and test outputs. This makes the job structure a potential **reproducible handoff contract** between model prototyping and integration in MATLAB/Simulink or another downstream environment.

The paper should not imply that the tool introduces a new neural algorithm, that every battery ML task is currently supported, or that a saved job is already a validated Simulink package. Instead, the contribution to test is whether this integrated, inspectable workflow lowers the practical burden of exploring battery regression models and reusing the selected model in a system-level study.

### Working title

**PyBattML: A workflow for exploring and integrating machine-learning battery models**

The title should be revised after resolving the project's naming. The repository uses PyBattML, VEstim, and the Python package name `vestim` in different places. Select a canonical name and explain any relationship among them. Keep “Simulink integration” in the title only after the connector is in the public repository and independently reproducible.

### Candidate abstract scaffold

Battery researchers increasingly use supervised machine learning to estimate quantities from battery cycling and drive-cycle measurements. Reproducible model development nevertheless requires coordinated data preparation, feature and target definition, architecture and hyperparameter exploration, repeated training, model selection, testing, and integration into a broader battery workflow. PyBattML is an interactive, locally executable environment for organizing these steps in job-based experiments. It supports configurable feedforward and recurrent regressors, feature augmentation and normalization, grid and Optuna search, repeated training, and saved task artifacts. We demonstrate a terminal-voltage estimation workflow on a documented battery dataset, select a model using validation data, and evaluate it on held-out conditions. We then use the saved job and selected task in a MATLAB/Simulink battery-estimation workflow, reporting prediction parity, system-level behavior, runtime, and limitations. PyBattML complements battery-specific ML platforms and general AutoML tools; it contributes an integrated researcher workflow and a reproducible model handoff rather than a new learning algorithm.

This abstract is conditional. Only retain the MATLAB/Simulink sentence after adding the connector, test data/permissions, and parity evidence to the public reproducibility materials. Add actual results and dataset/version details before submission.

## What the repository supports today

The application workflow covers data import, augmentation, hyperparameter configuration, training, and testing. Current implementation and example artifacts support the following:

- Import training, validation, and test files and organize a run into job/model/task/repetition directories.
- Select multiple input feature columns and one non-time target column.
- Prepare data with configured filtering, resampling, calculated columns/formulas, noise injection, normalization, and padding.
- Train FNN, LSTM, and GRU models through the GUI, with additional LSTM variants in lower-level code and sequence-to-sequence/whole-sequence paths.
- Set up grid-search configurations or Optuna search and run repeated training tasks.
- Persist checkpoints, task/job metadata, scalers, training summaries, and testing outputs so a selected model can be located and evaluated later.
- Test saved models and generate metrics, plots, and prediction tables.

The project owner reports jobs containing around 150 trained candidates, with repetition sweeps and later selection of a strong task. Treat this as a useful demonstrated scale point: include a shareable job manifest, candidate count, repetitions, elapsed time, hardware, failed runs, and the model-selection rule in the paper rather than relying on an isolated anecdote.

The GUI is currently **single-output**: it selects one target and the training setup sets `OUTPUT_SIZE=1`. A broad class of battery estimates fits this SISO pattern, for example measured signals/history to terminal voltage, SOC, SOH, temperature, capacity, or resistance, provided task-specific labels and alignment exist. Selecting a column in the UI is not evidence that each application has been validated. MIMO/co-estimation, such as predicting voltage and surface temperature jointly, is plausible but requires changes to target schema, data loading, loss scaling across physical units, missing-label treatment, per-target inverse scaling/metrics/plots, export metadata, and physical-consistency tests. It is not just increasing the final layer width.

The project shortcut is also a useful workflow component. The `.pbmlproj` file can define project-relative train/validation/test paths, file format, augmentation defaults, and launch behavior that opens data import. In the current workspace the sample project is a configuration template; it does not itself download a public dataset or contain a complete licensed data bundle. The installer has demo-data support, but that is not the same as an identified, citable open dataset. A research-ready project shortcut should include a dataset card/license, source/version/checksum, splits, feature/target configuration, and exact download or bundled-data instructions.

### Existing result evidence

The repository's offline FNN validation report records 24 LG NMC drive cycles (1,645,214 samples), prediction differences from saved repository outputs with a maximum absolute difference of 0.000501001 mV, and 12.450037 mV RMSE for 10 UDDS at 40 C [11]. It also reports a complete UDDS comparison of one FNN and one LSTM job. These are useful software parity and demonstration results, not a controlled cross-validation study, evidence of generalization to new cells/chemistries, or proof that the models outperform other methods.

## Why the job-to-MATLAB workflow is important

### User-reported MATLAB proof of concept

The project owner provided a MATLAB caller of the form:

```matlab
out = run_ptmi_ctvm_coulumb_counter( ...
    jobDir, modelTaskDir, caseCsv, ocvDir, true, 61.35, 'tsw', 'uniform');
```

The example passes both the parent job directory and selected model task, along with a measured case CSV and OCV data. It selects input-aligned or uniform-grid output and uses the fitted ML model within a larger coulomb-counter/voltage-model workflow. This illustrates the actual value proposition: PyBattML organizes the model exploration and preserves the winning task; MATLAB then uses that selected model alongside other battery-estimation logic and a battery/ECM simulation.

I could not locate the named `.m` connector in this repository snapshot. It should therefore be described as a **project-owner-reported external proof of concept**, not a repository-verified capability, until the source, requirements, test case, and prediction comparisons are included in the public project. The exact workflow deserves to be brought into the paper materials.

### What MathWorks already provides

MathWorks already has generic Simulink blocks for PyTorch and ONNX model prediction. The PyTorch Model Predict block runs a model through MATLAB's configured Python environment and provides preprocessing/postprocessing hooks; its current documentation says it was tested with Python 3.10/PyTorch 2.8 and does not support Rapid Accelerator [17]. The ONNX Model Predict block similarly uses Python `onnxruntime`, allows pre/postprocessing, and does not support Rapid Accelerator [18]. Imported PyTorch/ONNX networks can also be used with MATLAB's native network APIs and Simulink Predict block. Code generation is layer/toolbox-dependent; unsupported or generated custom layers can block code generation [19-21].

Therefore, PyBattML should **not** claim novelty for “running a neural network in Simulink.” The possible contribution is battery-job-aware export: carrying the trained model together with exact preprocessing, feature ordering, units, target scaling, sample-time assumptions, and recurrent state/reset semantics, with a Python-to-MATLAB parity report. This removes battery-specific translation work and makes the selected model traceable from a large search job into the system simulation.

### Integration levels

| Level | What it means | Evidence and constraints |
| --- | --- | --- |
| Python-backed Simulink coexecution | Run a PyTorch model via the PyTorch Model Predict block, or ONNX via ONNX Model Predict, with generated preprocessing and postprocessing. | Best initial route, especially for the FNN voltage case. Requires MATLAB/Simulink, the relevant toolbox/block, Python and PyTorch or ONNX Runtime. It is a host simulation workflow, not generated embedded inference; Rapid Accelerator is unsupported by the cited blocks. |
| Imported MATLAB network | Import a traced/exported PyTorch or ONNX graph into MATLAB and use it via a native Predict block. | Requires Deep Learning Toolbox and converter support packages. Layer conversion, tensor layout, custom layers, and network state need validation. |
| Code generation | Generate CPU/GPU code from an imported network whose layers support the target code-generation path. | Requires the relevant MATLAB Coder/GPU Coder capabilities. Custom-layer and target limitations apply. |
| Embedded/real-time deployment | Integrate and benchmark generated model code on a named target with timing/memory requirements. | A separate engineering and validation tier. Never infer real-time, safety, or production BMS readiness from a successful desktop Simulink simulation. |

### Current artifact gap and first prototype

For the included voltage FNN task, `task_info.json` records the ordered inputs `Power`, `Battery_Temp_degC`, `SOC`, `P_filter_2`, and `P_filter_0p2`, with target `Voltage`. The parent job has `job_metadata.json`, a MinMax scaler and normalized-column list, and `augmentation_metadata.json` describing causal Butterworth filters used to generate the filtered-power inputs. That is a strong starting point for a model manifest and preprocessing wrapper.

But the existing `.pt` artifacts are not a MATLAB interchange contract. `best_model.pth` is a PyTorch checkpoint dictionary; `best_model_export.pt` is another PyTorch pickle dictionary, not ONNX or MATLAB format. The training save routine writes a generic model-definition string even for an FNN, and the export dictionary lacks some architecture/preprocessing fields held in task/job metadata. The portable Python inference runner reconstructs supported models and reproduces prior FNN results closely, but MATLAB model conversion and Simulink execution have not been independently tested. Build the exporter from validated `task_info.json`, weights, scaler and augmentation metadata rather than treating the current `.pt` export as universal.

**Recommended first vertical slice:** the shipped FNN voltage job. Export the inference graph and a versioned manifest; implement the two causal filters, scaler transform over the right columns, feature order, clipped normalized output, and voltage inverse scaling; connect it through a PyTorch Model Predict block; replay a complete measured drive cycle; compare each sample with the Python reference; then connect the predictor to a minimal Simulink battery/ECM model and verify sample time and units. Add LSTM/GRU only after defining and testing recurrent state, start/reset, and warm-up semantics. The user-reported coulomb-counter/OCV POC can be the applied integration case once its connector is included and validated.

## Practical researcher journey to demonstrate

A compelling paper demonstration should show a complete, realistic workflow:

1. Open a `.pbmlproj` starter that points to a cited dataset's train/validation/test splits and supplies documented augmentation defaults.
2. Select signal inputs and one target, initially terminal voltage; treat SOC or temperature as separate tasks requiring verified labels and protocols.
3. Define a broad search space and run a meaningful batch of candidates/repetitions in a job, recording candidate count, failures, compute, elapsed time, and the search objective.
4. Select the model using validation data only. Reserve final test cells/cycles/conditions for unbiased evaluation; don't select the “best” model from test scores.
5. Preserve the exact task and parent job, including feature order, scaler, filters, configuration, and checkpoint.
6. Pass the selected task/job into the MATLAB coulomb-counter/OCV connector or Simulink system-model example; report Python/MATLAB parity, alignment mode, units, and system-level output.
7. Share the project shortcut, documented dataset acquisition/licensing details, selected job artifact, connector, and one-click reproduction instructions.

This is a stronger contribution than “a GUI trains LSTMs”: **a battery-ML experiment workbench that lets researchers search broadly, keep an auditable winner, and reuse that exact model in a separate battery-system workflow.** Prove its value with a workflow comparison or carefully measured task-completion/time study; don't assume the GUI itself guarantees faster work.

## Research questions and proposed evaluation

- **RQ1, workflow utility:** Does the integrated project/import/augmentation/search/job/testing flow reduce setup and experiment-management effort versus a documented script-based PyTorch baseline? Measure setup time, steps, failure points, and researcher feedback.
- **RQ2, scientific validity:** How do selected FNN/LSTM/GRU models perform across held-out cells, drive cycles, and temperatures under leakage-safe blocked splits? Report mean/spread over repetitions, per-condition error, bias, runtime, and compute.
- **RQ3, integration fidelity:** Does a job-derived model-plus-preprocessing bundle reproduce Python predictions in MATLAB/Simulink and preserve correct sample-time, units, filter initialization, and reset semantics?
- **RQ4, future MIMO utility:** What additional engineering and scientific validation are required for joint outputs such as voltage plus surface temperature? Keep this as future work unless the multi-output implementation and experiment are completed.

### Minimum scientific study

- Confirm exact public-dataset record, data version, download access, license, citation, and redistribution terms. The McMaster research catalog lists multi-temperature drive-cycle, fast-charge, aging, and state-estimation resources, but each dataset needs its own provenance and terms [9]. The 24 LG NMC files in the repository's validation materials are not by themselves evidence of an openly redistributable training set.
- Define one primary task precisely: e.g., instantaneous terminal voltage from measured signals/history, with causal input window, sample rate, sensor names, target alignment, and intended use.
- Split by complete cells, drive cycles, or conditions to test the intended generalization. Do not randomly split adjacent rows and report that as unseen-cell performance.
- Fit preprocessing using training data only; audit whether filters are causal or use future samples. Preserve scaler/filter metadata and verify that testing uses the same path.
- Compare FNN/LSTM/GRU with matched inputs and compute. If the ~150-model search is used, show the search budget, repetitions, validation selection rule, and untouched final test set.
- Use fair baselines: simple linear/persistence regression, AutoGluon Tabular on equivalent rows/features, and a calibrated ECM/physics-based reference where its parameters and assumptions are defensible. BatteryML is a fair comparator only on a matched task/label/split, not by transplanting its RUL/SOH/SOC leaderboard numbers to voltage regression.
- Report voltage MAE/RMSE in mV, bias, high-percentile/max error, condition-specific results, inference time, and memory. For SOC/temperature tasks define task-specific labels, state baselines, lag/uncertainty analysis, and units.

### Deployment-interoperability study

Use one selected FNN task as the first vertical slice. Run the same raw sequence through the Python runner and the Simulink PyTorch Model Predict block plus generated preprocessing/postprocessing; compare every output in physical units within a justified tolerance. Then connect it to a small battery/ECM Simulink workflow such as the project owner's coulomb-counter/OCV example, and compare input-aligned versus uniform-grid outputs. Record MATLAB release, required toolboxes/blocks, Python/PyTorch version, preprocessing hashes, elapsed time, and any limitations. Treat this as host coexecution, not codegen or embedded deployment [17]. Extend to recurrent models only after state and reset parity tests.

## Related tools: practical comparison

| Tool | Practical strength | Default task focus and gap relative to this workflow | Positioning |
| --- | --- | --- | --- |
| [BatteryML](https://github.com/microsoft/BatteryML) [3] | BatteryData format; public/cycler preprocessing; extensible feature/label/model modules; classical and neural baselines; published RUL, SOH, and SOC benchmarks. | Its paper does not provide a ready-configured pointwise terminal-voltage or surface-temperature trajectory benchmark, nor a job-to-Simulink workflow. It is extensible, so do not claim these tasks are impossible without substantial adaptation. | Closest battery-ML prior art. Differentiate through interactive GUI/job workflow and task-to-system-model handoff, not “first battery ML platform.” Its GitHub repository was archived 15 Sep 2026; its published paper remains prior art. |
| [AutoGluon](https://auto.gluon.ai/stable/) [4, 5] | Strong local AutoML for tabular regression and probabilistic time-series forecasting, including model selection, ensembling, and tuning. | No battery cycler conventions or curated battery target/augmentation setup. Tabular regression is a fair baseline for instantaneous voltage if inputs/splits match; forecasting is a different framing unless horizon/covariates match. | Generic comparator. Do not claim data privacy superiority because cloud routes are optional and local AutoGluon is available. |
| [BEEP](https://github.com/TRI-AMDD/beep) [6] | Structures/validates several cycler formats, extracts battery features, supports battery evaluation and early cycle-life prediction. | Not a general GUI workbench for user-selected signal-to-scalar FNN/LSTM/GRU regression or model handoff to Simulink, based on cited project/paper. | Complementary battery data/lifecycle workflow; stronger cycler-format support. |
| [PyBaMM](https://pybamm.org/) [1] | Define and simulate electrochemical battery models, experiments, parameters, solvers, and outputs such as terminal voltage. | Not a generic CSV-driven neural-model search/training GUI. | Physics-based reference and potential integration for simulation-generated data or baseline comparison, not a direct substitute. |
| [MathWorks Simulink prediction blocks](https://www.mathworks.com/help/deeplearning/ug/interoperability-between-deep-learning-toolbox-tensorflow-pytorch-and-onnx.html) [17-21] | General PyTorch/ONNX model execution in Simulink, import into MATLAB networks, and conditional code generation. | No battery-job parser that reconstructs PyBattML scaler, filters, feature order, target semantics, search provenance, and parity checks as one reproducible unit. | Direct connector baseline. The proposed PyBattML value is job-aware packaging and validation, not reimplementing generic inference blocks. |

**Adjacent:** PyBOP parameterizes and optimizes battery models, including PyBaMM models; it solves a different modeling objective than supervised neural regression on imported datasets [2].

### Privacy is a property to validate, not a novelty claim

Local training can be useful when lab data cannot leave an institution, but local execution is available in AutoGluon and the other open tools too. A targeted source scan found package/setup network calls and a localhost conversion request, but no obvious remote data upload in the inspected training path; this was not a complete audit. Before making a privacy claim, test a prepared installation with outbound access blocked, trace connections during import/training/testing, and audit dependencies and model downloads. State precisely what runs locally and what may connect externally during setup. Avoid “guarantees data privacy.”

## Applied-journal and software-paper strategy

| Rank | Journal | When it fits this work | What the manuscript must demonstrate |
| --- | --- | --- | --- |
| 1 | [Journal of Energy Storage](https://www.sciencedirect.com/journal/journal-of-energy-storage/about/aims-and-scope) | Best applied target if centered on battery prediction, evaluation, and the model's role in a battery simulation. Its scope includes storage modeling, testing, EV applications, management, and control. | A substantive battery-centered result, held-out cells/cycles/conditions, fair ML and physical baselines, reproducible data/code, and validated handoff. A tool description or optimizer alone is insufficient; its guide says storage must be the central scientific contribution. |
| 2 | [Journal of Power Sources](https://www.sciencedirect.com/journal/journal-of-power-sources/about/aims-and-scope) | Strong battery-device/diagnostics audience; its priorities include experimentally validated AI/ML predictive modeling. | More than software: multi-cell/condition validation, electrochemical interpretation, and evidence the model/system integration improves or informs battery diagnostics/performance. A verified Simulink system-model study could strengthen fit. |
| 3 | [Applied Energy](https://www.sciencedirect.com/journal/applied-energy/about/aims-and-scope) | Stretch target if the work addresses an energy-system question rather than only a cell estimator. | System-level consequence such as BMS control, fast-charge decisions, usable energy, safety, pack thermal management, or EV operation. |

**Recommendation:** If the connector and system case are validated, prepare an applied paper for the *Journal of Energy Storage*; consider *Journal of Power Sources* if experimental and electrochemical implications are strong. If the software remains the main contribution and the connector is a roadmap item, target SoftwareX or JOSS [7, 8]. Recheck current scopes/instructions at submission. *Journal of Energy Storage* currently requires a data-availability statement/research-data linking; Elsevier journals require generative-AI disclosure where applicable [12-14].

## Manuscript structure and readiness

1. Need: friction in battery-ML experimentation and model reuse; identify user and task scope.
2. State of field: BatteryML, AutoGluon, BEEP, PyBaMM, PyBOP, and MathWorks interoperability; build-versus-contribute rationale.
3. Software design: project shortcut, augmentation, search/task hierarchy, artifact provenance, model exporter, Simulink connector, limitations.
4. Demonstration: public dataset, task/split, search budget, selected candidate, Python-vs-MATLAB parity, system-model result.
5. Impact and limitations: researcher workflow evidence, generalization, supported target/model types, MIMO/deployment roadmap.

Before submission:

- Resolve canonical product/package identity: PyBattML, VEstim, and `vestim`.
- Add the externally described MATLAB connector and requirements to a versioned public repository; document which export formats and model families it truly supports.
- Define a versioned job export manifest separate from internal checkpoint serialization. Include feature order, target(s)/units, scaler, augmentation, sample time, state/reset behavior, checkpoint hash, and library versions.
- Validate the FNN-to-Simulink vertical slice; then validate LSTM/GRU state/reset/warm-up behavior separately.
- If MIMO is implemented, specify multi-target loss scaling, label masks, inverse scaling, metrics, output physical units, and coupled-system consistency checks.
- Publish a complete root README, install instructions, public release, tests/CI, contribution/support path, and a DOI-backed software archive.
- Confirm dataset-level access, citation, license, and reuse terms; make the starter project self-contained or provide exact download/split instructions.
- Report model-search scale, validation-based selection, untouched test results, repetitions, runtime, hardware, and uncertainty. Do not pick the winner using final test labels.
- Keep local/offline claims proportional to an actual network-egress audit.

## References and research links

1. Sulzer, V., Marquis, S. G., Timms, R., Robinson, M., and Chapman, S. J. (2021). “Python Battery Mathematical Modelling (PyBaMM).” *Journal of Open Research Software*, 9(1), 14. https://doi.org/10.5334/jors.309.
2. Planden, B., Courtier, N. E., Robinson, M., Khetarpal, A., Planella, F. B., and Howey, D. A. (2025). “PyBOP: A Python package for battery model optimisation and parameterisation.” *Journal of Open Source Software*, 10(116), 7874. https://doi.org/10.21105/joss.07874.
3. Zhang, H., Gui, X., Zheng, S., Lu, Z., Li, Y., and Bian, J. (2024). “BatteryML: An Open-source Platform for Machine Learning on Battery Degradation.” *Proceedings of the Twelfth International Conference on Learning Representations (ICLR 2024)*. [arXiv:2310.14714](https://arxiv.org/abs/2310.14714); [ICLR record](https://iclr.cc/virtual/2024/poster/17628); [code](https://github.com/microsoft/BatteryML).
4. Erickson, N., Mueller, J., Shirkov, A., Zhang, H., Larroy, P., Li, M., and Smola, A. (2020). “AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data.” arXiv:2003.06505. https://doi.org/10.48550/arXiv.2003.06505.
5. Shchur, O., Turkmen, C., Erickson, N., Shen, H., Shirkov, A., Hu, T., and Wang, Y. (2023). “AutoGluon-TimeSeries: AutoML for Probabilistic Time Series Forecasting.” *International Conference on Automated Machine Learning*. arXiv:2308.05566. https://doi.org/10.48550/arXiv.2308.05566.
6. Herring, P. et al. (2020). “BEEP: A Python library for Battery Evaluation and Early Prediction.” *SoftwareX*, 11, 100506. https://doi.org/10.1016/j.softx.2020.100506.
7. [JOSS paper format and author requirements](https://joss.readthedocs.io/en/latest/paper.html).
8. [JOSS review criteria](https://joss.readthedocs.io/en/latest/review_criteria.html).
9. [SoftwareX aims and scope](https://www.sciencedirect.com/journal/softwarex/about/aims-and-scope).
10. McMaster Battery Research Group. [Dataset and Algorithms](https://battery.mcmaster.ca/research/datasets-and-algorithms/). Cite the selected dataset's own record, version, and reuse terms in the manuscript.
11. Repository-local. [Offline inference validation report](../examples/paper_reproducibility/VALIDATION.md), validation recorded 21-22 September 2026. Software parity evidence, not a dataset citation or cross-validation study.
12. [Journal of Energy Storage aims and scope](https://www.sciencedirect.com/journal/journal-of-energy-storage/about/aims-and-scope) and [author guide](https://www.sciencedirect.com/journal/journal-of-energy-storage/publish/guide-for-authors).
13. [Journal of Power Sources aims and scope](https://www.sciencedirect.com/journal/journal-of-power-sources/about/aims-and-scope) and [author guide](https://www.sciencedirect.com/journal/journal-of-power-sources/publish/guide-for-authors).
14. [Applied Energy aims and scope](https://www.sciencedirect.com/journal/applied-energy/about/aims-and-scope).
15. Chemali, E., Kollmeyer, P. J., Preindl, M., and Emadi, A. (2018). “State-of-charge estimation of Li-ion batteries using deep neural networks: A machine learning approach.” *Journal of Power Sources*, 400, 242-255. https://doi.org/10.1016/j.jpowsour.2018.06.104.
16. Severson, K. A. et al. (2019). “Data-driven prediction of battery cycle life before capacity degradation.” *Nature Energy*, 4, 383-391. https://doi.org/10.1038/s41560-019-0356-8.
17. MathWorks. [PyTorch Model Predict block](https://www.mathworks.com/help/deeplearning/ref/pytorchmodelpredict.html). Python coexecution path; current page lists tested Python/PyTorch versions and Rapid Accelerator limitation.
18. MathWorks. [ONNX Model Predict block](https://www.mathworks.com/help/deeplearning/ref/onnxmodelpredict.html). ONNX Runtime coexecution with preprocessing/postprocessing hooks; Rapid Accelerator limitation.
19. MathWorks. [Interoperability between Deep Learning Toolbox, TensorFlow, PyTorch, and ONNX](https://www.mathworks.com/help/deeplearning/ug/interoperability-between-deep-learning-toolbox-tensorflow-pytorch-and-onnx.html).
20. MathWorks. [`importNetworkFromPyTorch`](https://www.mathworks.com/help/deeplearning/ref/importnetworkfrompytorch.html). Converter add-on, supported formats, custom-layer and code-generation limitations.
21. MathWorks. [Code Generation for Deep Learning Networks](https://www.mathworks.com/help/deeplearning/ug/code-generation-for-deep-learning-networks.html).

## Source and scope note

Repository capability statements were checked against the user guide, GUI target/model setup, training manager, augmentation manager, project-file configuration, model-save code, portable inference README, and validation report. The MATLAB coulomb-counter/OCV caller described above was supplied by the project owner but is not present in this repository snapshot; its source and output parity need to be archived and reviewed before publication. MathWorks compatibility statements were checked against official documentation on 28 September 2026. This remains a focused positioning review, not a systematic review; conduct and document broader literature searches before making exhaustive novelty claims.
