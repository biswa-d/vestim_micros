"""Portable temperature-dependent 1RC ECM; no MATLAB or vestim dependency.

Port of Highest_Final_RC_With_Temp_6Temps_1000h_Benchmark.m. Parameters are
read from six small MAT files. Historical paper exports are validation data,
never inference inputs. See ECM_README.md for preprocessing and parity limits.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.interpolate import interp1d, RegularGridInterpolator
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ECMModel:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.config = json.loads((self.directory / "ecm_config.json").read_text(encoding="utf-8"))
        if self.config["model_type"] != "ECM_1RC":
            raise ValueError("Only ECM_1RC is supported")
        if self.config.get("current_convention") != "negative_discharge":
            raise ValueError("ECM expects negative current for discharge")
        tables = {}
        for entry in self.config["parameters"]:
            path = (self.directory / entry["path"]).resolve()
            if not path.is_relative_to(self.directory):
                raise ValueError("ECM parameters must be inside their model folder")
            data = loadmat(path)
            par = [v for k, v in data.items() if not k.startswith("__") and getattr(v, "shape", None) == (15, 3)]
            soc = [v.ravel().astype(float) for k, v in data.items() if "soc" in k.lower() and np.size(v) == 15]
            ocv = [v.ravel().astype(float) for k, v in data.items() if "ocv" in k.lower() and np.size(v) == 15]
            if len(par) != 1 or len(soc) != 1 or len(ocv) != 1:
                raise ValueError(f"Ambiguous/missing R0,R1,C1/SOC/OCV tables: {path}")
            order = np.argsort(soc[0])
            bp = soc[0][order]
            values = np.column_stack([par[0][order].astype(float), ocv[0][order]])
            if not np.isfinite(values).all() or not np.isfinite(bp).all() or np.any(np.diff(bp) <= 0):
                raise ValueError(f"Invalid parameter table: {path}")
            temperature = float(entry["temperature_c"])
            if temperature in tables:
                raise ValueError("Duplicate parameter temperature")
            tables[temperature] = (bp, values)
        self.temperatures = np.array(sorted(tables))
        if len(tables) < 2:
            raise ValueError("At least two temperature tables are required")
        self.soc = tables[float(self.config["master_soc_temperature_c"])][0]
        grids = []
        for temperature in self.temperatures:
            bp, values = tables[temperature]
            grid = interp1d(bp, values, axis=0, bounds_error=False, fill_value="extrapolate")(self.soc)
            grid[:, :3] = np.maximum(grid[:, :3], np.finfo(float).eps)
            grid[:, 2] = np.log(np.maximum(grid[:, 2], np.finfo(float).tiny))
            grids.append(grid)
        self.lookup = RegularGridInterpolator((self.soc, self.temperatures), np.stack(grids, axis=1))

    def predict(self, soc_percent, temperature, current, time):
        soc, temp, current, time = [np.asarray(v, dtype=float) for v in (soc_percent, temperature, current, time)]
        if any(v.ndim != 1 or len(v) != len(time) or not np.isfinite(v).all() for v in (soc, temp, current, time)) or not len(time):
            raise ValueError("ECM requires finite, equally sized one-dimensional inputs")
        query = np.column_stack([np.clip(soc, self.soc.min(), self.soc.max()),
                                 np.clip(temp, self.temperatures.min(), self.temperatures.max())])
        r0, r1, log_c1, ocv = self.lookup(query).T
        c1 = np.exp(log_c1)
        rc = np.zeros(len(time))
        rc[0] = float(self.config.get("initial_rc_voltage_v", 0))
        dt = np.diff(time)
        dt = np.where(np.isfinite(dt) & (dt > 0), dt, 1.0)
        tau = np.maximum(np.maximum(r1[:-1], np.finfo(float).eps) *
                         np.maximum(c1[:-1], np.finfo(float).eps), 1e-12)
        decay = np.exp(-dt / tau)
        for k in range(1, len(time)):
            rc[k] = decay[k-1] * rc[k-1] + max(r1[k-1], np.finfo(float).eps) * (1-decay[k-1]) * current[k-1]
        return ocv + r0 * current + rc

    def prepare(self, frame):
        aliases = {
            "Voltage": ("Voltage", "Voltage(V)", "Voltage_V_"),
            "Current": ("Current", "Current(A)", "Current_A_"),
            "SOC": ("SOC",), "Battery_Temp_degC": ("Battery_Temp_degC",),
        }
        values = {}
        for target, names in aliases.items():
            column = next((name for name in names if name in frame), None)
            if column is None:
                raise ValueError(f"ECM requires {target}; accepted column names: {names}")
            values[target] = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        time_col = next((name for name in ("Adjusted_Test_Time(s)", "Adjusted_Test_Time_s_") if name in frame), None)
        if time_col:
            time = pd.to_numeric(frame[time_col], errors="raise").to_numpy(dtype=float)
            data = pd.DataFrame({"Time_s": time, **values})
            data = data.loc[np.isfinite(data.to_numpy()).all(axis=1)].drop_duplicates("Time_s", keep="first")
            if len(data) < 2 or np.any(np.diff(data.Time_s) <= 0):
                raise ValueError("Adjusted test time must increase after duplicate removal")
            time = data.Time_s.to_numpy()
            # MATLAB t_start:1:t_end; no extrapolated extra endpoint.
            uniform = time[0] + np.arange(int(np.floor(time[-1] - time[0])) + 1, dtype=float)
            result = pd.DataFrame({name: np.interp(uniform, time, data[name]) for name in values})
            result.insert(0, "Time_s", uniform - uniform[0])
            policy = "adjusted_time_resampled_1s"
        else:
            # ML raw CSV Time can reset between test segments. The paper exports
            # use one second per row, so never guess an elapsed axis from that column.
            result = pd.DataFrame(values)
            if result.empty or not np.isfinite(result.to_numpy()).all():
                raise ValueError("Raw ECM inputs must be nonempty and finite")
            step = float(self.config["sample_interval_s"])
            if not np.isfinite(step) or step <= 0:
                raise ValueError("sample_interval_s must be positive")
            result.insert(0, "Time_s", np.arange(len(result)) * step)
            policy = "sample_interval_from_ecm_config"
        if result.SOC.max() <= 1.5:
            result["SOC"] *= 100
        return result, policy

    def run(self, test_file, bias_correct=False):
        frame, policy = self.prepare(pd.read_csv(test_file))
        actual = frame.Voltage.to_numpy()
        raw = self.predict(frame.SOC, frame.Battery_Temp_degC, frame.Current, frame.Time_s)
        bias = float(np.mean(raw - actual))
        predicted = raw - bias if bias_correct else raw.copy()
        error = predicted - actual
        return dict(predictions=predicted, true_values=actual, raw_predictions=raw,
                    frame=frame, time_s=frame.Time_s.to_numpy(), time_policy=policy,
                    rmse=float(np.sqrt(mean_squared_error(actual, predicted))*1000),
                    mae=float(mean_absolute_error(actual, predicted)*1000),
                    r2=float(r2_score(actual, predicted)), bias_correction_applied=bias_correct,
                    raw_bias_mv=bias*1000, raw_rmse_mv=float(np.sqrt(np.mean((raw-actual)**2))*1000),
                    max_abs_error_mv=float(np.max(np.abs(error))*1000))
