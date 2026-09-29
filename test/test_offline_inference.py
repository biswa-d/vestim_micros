"""Repository parity tests; the distributed runner itself does not import vestim."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXAMPLE = ROOT / "examples" / "paper_reproducibility"
spec = importlib.util.spec_from_file_location("offline", EXAMPLE / "run_offline_inference.py")
offline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(offline)
from vestim.services.model_training.src.FNN_model import FNNModel
from vestim.services.model_training.src.LSTM_model import LSTMModel
from vestim.services.model_training.src.GRU_model import GRUModel
from vestim.services.model_testing.src.testing_service import apply_inference_filter
from vestim.services.model_testing.src.continuous_testing_service import ContinuousTestingService


class OfflineTests(unittest.TestCase):
    def test_model_checkpoint_and_state_parity(self):
        torch.manual_seed(12)
        with tempfile.TemporaryDirectory() as directory:
            for kind, ref in [
                ("FNN", FNNModel(2, 1, [4, 3], .1, True)),
                ("LSTM", LSTMModel(2, 4, 2, "cpu", .1)),
                ("GRU", GRUModel(2, 4, 2, dropout_prob=.1, apply_clipped_relu=True)),
                ("GRU", GRUModel(2, 4, 2, use_layer_norm=True, apply_clipped_relu=True)),
            ]:
                ref.eval()
                path = Path(directory) / "best_model.pth"
                torch.save({"model_state_dict": ref.state_dict()}, path)
                hp = dict(INPUT_SIZE=2, HIDDEN_LAYER_SIZES=[4, 3], HIDDEN_UNITS=4,
                          LAYERS=2, DROPOUT_PROB=.1, normalization_applied=True,
                          GRU_USE_LAYERNORM=getattr(ref, "use_layer_norm", False))
                task = dict(hyperparams=hp, model_type=kind, best_model_path=str(path),
                            job_metadata={}, data_loader_params={"feature_columns": ["x", "z"]})
                model = offline._load_model(task, torch.device("cpu"))
                a = b = None
                for x in torch.randn(5, 1, 1, 2):
                    if kind == "FNN":
                        y, expected = model(x[:, 0]), ref(x[:, 0])
                    elif kind == "GRU":
                        y, a = model(x, a)
                        expected, b = ref(x, b)
                    else:
                        y, a = model(x, *(a or (None, None)))
                        expected, b = ref(x, *(b or (None, None)))
                    torch.testing.assert_close(y, expected, rtol=0, atol=0)

    def test_prediction_filters(self):
        values = np.linspace(-1, 2, 60, dtype=np.float32) ** 2
        for kind in ["None", "Moving Average", "Exponential Moving Average", "Savitzky-Golay"]:
            hp = dict(INFERENCE_FILTER_TYPE=kind, INFERENCE_FILTER_WINDOW_SIZE="8",
                      INFERENCE_FILTER_ALPHA="0.2", INFERENCE_FILTER_POLYORDER="2")
            np.testing.assert_array_equal(offline._filter_predictions(values, hp),
                                          apply_inference_filter(values, {"hyperparams": hp}))

    def test_bundled_job_against_repository_service(self):
        job = next(EXAMPLE.glob("job_*"))
        task = offline._build_task(job, offline._find_model_dir(job, None))
        raw_path = EXAMPLE / "LG_NMC_test_data" / "10_UDDS_40C.csv"
        if not raw_path.exists():
            self.skipTest("Raw test data is distributed separately")
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.csv"
            pd.read_csv(raw_path).head(2000).to_csv(raw, index=False)
            scaler = offline._load_scaler(task)
            frame = offline._apply_augmentation(pd.read_csv(raw), job)
            cols = list(scaler.feature_names_in_)
            frame[cols] = scaler.transform(frame[cols])
            processed = Path(directory) / "processed.csv"
            frame.to_csv(processed, index=False)
            model = offline._load_model(task, torch.device("cpu"))
            actual = offline._run_inference(model, "FNN", scaler, task, raw, 0, torch.device("cpu"))
            service = ContinuousTestingService("cpu")
            # Avoid old torch.load default incompatibility in the repository service.
            service.model_instance = FNNModel(5, 1, [90, 45], .02, True).eval()
            service.model_instance.load_state_dict(model.state_dict())
            service.scaler = scaler
            with contextlib.redirect_stdout(io.StringIO()):
                expected = service.run_continuous_testing(task, task["best_model_path"], str(processed), warmup_samples=0)
            self.assertIsNotNone(expected)
            np.testing.assert_allclose(actual["predictions"], expected["predictions"], rtol=0, atol=1e-6)
            self.assertAlmostEqual(actual["rmse"], expected["rms_error_mv"], places=4)

    def test_configuration_failures_and_warmup(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            (job / "augmentation_metadata.json").write_text(json.dumps(
                {"resampling": {"applied": True, "frequency": "1Hz"}, "applied_filters": []}))
            with self.assertRaisesRegex(NotImplementedError, "resampling"):
                offline._apply_augmentation(pd.DataFrame({"x": [1.0]}), job)
            for name in ["repeat_1", "repeat_2"]:
                task = job / "models" / name
                task.mkdir(parents=True)
                (task / "task_info.json").write_text("{}")
                (task / "best_model.pth").touch()
            with self.assertRaisesRegex(ValueError, "Multiple models"):
                offline._find_model_dir(job, None)
        task = dict(model_type="LSTM", hyperparams={"LOOKBACK": "17"}, data_loader_params={})
        self.assertEqual(offline._warmup_samples(task, None), 17)
        self.assertEqual(offline._warmup_samples(task, 5), 5)
        task["model_type"] = "FNN"
        self.assertEqual(offline._warmup_samples(task, None), 0)

    def test_portable_multi_job_cli_and_missing_scaler(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copy2(EXAMPLE / "run_offline_inference.py", root)
            data = pd.DataFrame({"x": np.linspace(0, 1, 30), "Voltage": np.linspace(3, 4, 30)})
            data.to_csv(root / "test.csv", index=False)
            for name in ["job_A", "job_B"]:
                job = root / name
                model_dir = job / "models" / "FNN" / "repeat"
                model_dir.mkdir(parents=True)
                (job / "scalers").mkdir()
                scaler = MinMaxScaler().fit(data)
                joblib.dump(scaler, job / "scalers" / "augmentation_scaler.joblib")
                (job / "job_metadata.json").write_text(json.dumps(dict(normalization_applied=True,
                    scaler_path="scalers" + chr(92) + "augmentation_scaler.joblib")))
                hp = dict(MODEL_TYPE="FNN", INPUT_SIZE=1, HIDDEN_LAYER_SIZES=[3],
                          normalization_applied=True, FEATURE_COLUMNS=["x"], TARGET_COLUMN="Voltage")
                (model_dir / "task_info.json").write_text(json.dumps({"hyperparams": hp}))
                torch.save({"model_state_dict": FNNModel(1, 1, [3], apply_clipped_relu=True).state_dict()},
                           model_dir / "best_model.pth")
            command = [sys.executable, "-I", "run_offline_inference.py", "--job-dir", "job_A", "job_B",
                       "--test-file", "test.csv", "--output-dir", "results"]
            run = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertEqual(len(pd.read_csv(root / "results/comparison_metrics.csv")), 2)
            self.assertTrue((root / "results/comparison_test_Voltage.png").is_file())
            self.assertTrue((root / "results/job_A/test/prediction_plot.png").is_file())
            (root / "job_A/scalers/augmentation_scaler.joblib").unlink()
            run = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("required scaler missing", run.stderr)


if __name__ == "__main__":
    unittest.main()
