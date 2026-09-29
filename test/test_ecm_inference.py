"""Portable ECM checks against analytical dynamics and saved MATLAB metrics."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
from scipy.io import savemat
ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/paper_reproducibility"
sys.path.insert(0, str(EXAMPLE))
from ecm_inference import ECMModel


class ECMTests(unittest.TestCase):
    def test_constant_parameters_match_analytical_step_response(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            entries = []
            for temperature in (0, 40):
                name = f"table_{temperature}.mat"
                savemat(folder / name, dict(SOC_HPPC=np.linspace(0, 100, 15),
                    OCV_HPPC=np.full(15, 4.0), par_table=np.tile([.01, .02, 1000.], (15, 1))))
                entries.append(dict(temperature_c=temperature, path=name))
            config = dict(model_type="ECM_1RC", current_convention="negative_discharge",
                          master_soc_temperature_c=0, parameters=entries, sample_interval_s=1)
            (folder / "ecm_config.json").write_text(json.dumps(config))
            model = ECMModel(folder)
            time = np.arange(100.)
            voltage = model.predict(np.full(100, 50.), np.full(100, 20.), np.full(100, -2.), time)
            expected = 4.0 - .02 - .04 * (1 - np.exp(-time / 20.))
            np.testing.assert_allclose(voltage, expected, rtol=0, atol=1e-14)
            # Each call initializes its own RC state.
            np.testing.assert_array_equal(voltage, model.predict(np.full(100, 50.), np.full(100, 20.), np.full(100, -2.), time))

    def test_original_matlab_benchmark_cases(self):
        original = EXAMPLE / "ECM_LG_NMC_Test_Time"
        metrics = original / "Results/All_1RC_6Temps_Inference/PerFile_Inference_Timing.csv"
        if not metrics.exists():
            self.skipTest("Original benchmark inputs are development-only")
        model = ECMModel(EXAMPLE / "ecm_1rc_lg_nmc")
        for row in pd.read_csv(metrics).itertuples():
            with self.subTest(file=row.File):
                path, = original.glob("*C_DCH_Only/" + row.File)
                result = model.run(path)
                self.assertEqual(len(result["predictions"]), row.Samples)
                self.assertAlmostEqual(result["rmse"], row.RMSE_mV, places=7)
                self.assertAlmostEqual(result["raw_bias_mv"], row.Bias_mV, places=7)

    def test_isolated_cli_and_explicit_bias_correction(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for name in ("run_offline_inference.py", "ecm_inference.py"):
                shutil.copy2(EXAMPLE / name, folder / name)
            shutil.copytree(EXAMPLE / "ecm_1rc_lg_nmc", folder / "ecm")
            data = pd.DataFrame(dict(Voltage=np.full(30, 4.), Current=np.full(30, -1.),
                                    SOC=np.linspace(1, .99, 30), Battery_Temp_degC=np.full(30, 25.)))
            data.to_csv(folder / "10_UDDS_40C.csv", index=False)
            command = [sys.executable, "-I", "run_offline_inference.py", "--ecm-dir", "ecm",
                       "--test-file", "10_UDDS_40C.csv", "--ecm-bias-correction",
                       "--paper-window", "--output-dir", "results"]
            run = subprocess.run(command, cwd=folder, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            output = pd.read_csv(folder / "results/10_UDDS_40C/predictions.csv")
            self.assertIn("Raw_Predicted_Voltage", output)
            self.assertAlmostEqual(float((output.Predicted_Voltage-output.True_Voltage).mean()), 0, places=12)
            self.assertTrue((folder / "results/10_UDDS_40C/prediction_plot.png").exists())
            self.assertTrue(pd.read_csv(folder / "results/comparison_metrics.csv").bias_correction_applied.iloc[0])

    def test_adjusted_time_resampling_alignment(self):
        model = ECMModel(EXAMPLE / "ecm_1rc_lg_nmc")
        frame = pd.DataFrame({"Adjusted_Test_Time(s)": [1., 1., 3.],
                              "Voltage(V)": [4., 99., 3.], "Current(A)": [-1., 99., -3.],
                              "SOC": [1., 1., .9], "Battery_Temp_degC": [25., 99., 27.]})
        prepared, policy = model.prepare(frame)
        np.testing.assert_array_equal(prepared.Time_s, [0, 1, 2])
        np.testing.assert_array_equal(prepared.Voltage, [4., 3.5, 3.])
        self.assertEqual(policy, "adjusted_time_resampled_1s")


if __name__ == "__main__":
    unittest.main()
