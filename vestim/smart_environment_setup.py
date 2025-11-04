#!/usr/bin/env python3
"""
Smart Environment Setup for Vestim
Detects system capabilities and installs appropriate dependencies
"""

import subprocess
import sys
import os
import json
import tempfile
import time
import shutil
from pathlib import Path
from typing import Optional, Tuple, Dict, List


class SmartEnvironmentSetup:
    """Intelligent environment setup that adapts to target system capabilities"""

    def __init__(self, install_dir: str, project_dir: str = None, log_callback=None):
        self.install_dir = Path(install_dir)
        self.project_dir = Path(project_dir) if project_dir else self.install_dir / "project"
        self.python_dir = self.install_dir / "python"
        self.venv_dir = self.install_dir / "venv"
        self.log_file = self.install_dir / "setup.log"
        self.config_file = self.install_dir / "environment_config.json"
        self.log_callback = log_callback or print

        # Environment state
        self.system_info = {}
        self.install_config = {}
        
    def log(self, message: str, level: str = "INFO"):
        """Log message to file and callback"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        
        # Ensure install directory exists before writing to log
        self.install_dir.mkdir(parents=True, exist_ok=True)
        
        # Write to file
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + "\n")
        
        # Call callback
        self.log_callback(log_entry)
        
    def detect_system_capabilities(self) -> Dict:
        """Detect system GPU, CUDA, and Python capabilities"""
        self.log("=== System Capability Detection ===")
        
        # Initialize detection flags and variables
        cuda_detected = False
        driver_version = None
        gpu_name = None
        
        capabilities = {
            "has_nvidia_gpu": False,
            "cuda_version": None,
            "cuda_major": None,
            "cuda_minor": None,
            "gpu_name": None,
            "python_version": sys.version,
            "architecture": "x64" if sys.maxsize > 2**32 else "x86",
            "platform": sys.platform
        }
        
        # Detect NVIDIA GPU and CUDA
        # Try nvidia-ml-py3 first (more reliable, but might not be installed)
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            if device_count > 0:
                capabilities["has_nvidia_gpu"] = True
                # Get GPU name
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                capabilities["gpu_name"] = gpu_name
                self.log(f"NVIDIA GPU detected via pynvml: {gpu_name}")
                
                # Get CUDA version from driver
                cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
                major = cuda_version // 1000
                minor = (cuda_version % 1000) // 10
                capabilities["cuda_major"] = major
                capabilities["cuda_minor"] = minor
                capabilities["cuda_version"] = f"{major}.{minor}"
                cuda_detected = True
                self.log(f"CUDA Driver Version: {major}.{minor}")
                return capabilities
                
        except ImportError:
            self.log("pynvml not available, trying nvidia-smi...")
        except Exception as e:
            self.log(f"pynvml detection failed: {e}, trying nvidia-smi...")
        
        # Fallback to nvidia-smi if pynvml failed or is not available
        if not capabilities.get("has_nvidia_gpu", False):
            try:
                result = subprocess.run([
                    "nvidia-smi", "--query-gpu=name,driver_version", 
                    "--format=csv,noheader"
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if lines:
                        gpu_info = lines[0].split(', ')
                        if len(gpu_info) >= 2:
                            capabilities["has_nvidia_gpu"] = True
                            capabilities["gpu_name"] = gpu_info[0]
                            driver_version = gpu_info[1]
                            self.log(f"NVIDIA GPU detected via nvidia-smi: {gpu_info[0]}, Driver: {driver_version}")
                            
                            # Try to get CUDA version directly
                            cuda_result = subprocess.run([
                                "nvidia-smi", "--query-gpu=cuda_version", 
                                "--format=csv,noheader"
                            ], capture_output=True, text=True, timeout=10)
                            
                            cuda_detected = False
                            if cuda_result.returncode == 0:
                                cuda_str = cuda_result.stdout.strip()
                                if cuda_str and cuda_str != "N/A":
                                    parts = cuda_str.split('.')
                                    if len(parts) >= 2:
                                        capabilities["cuda_major"] = int(parts[0])
                                        capabilities["cuda_minor"] = int(parts[1])
                                        capabilities["cuda_version"] = cuda_str
                                        self.log(f"CUDA Version detected: {cuda_str}")
                                        cuda_detected = True
                            
                            # If direct CUDA detection failed, try nvcc
                            if not cuda_detected:
                                try:
                                    nvcc_result = subprocess.run([
                                        "nvcc", "--version"
                                    ], capture_output=True, text=True, timeout=10)
                                    
                                    if nvcc_result.returncode == 0:
                                        # Parse nvcc output for version
                                        lines = nvcc_result.stdout.split('\n')
                                        for line in lines:
                                            if 'release' in line.lower():
                                                parts = line.split()
                                                for i, part in enumerate(parts):
                                                    if part.lower() == 'release' and i + 1 < len(parts):
                                                        version_str = parts[i + 1].rstrip(',')
                                                        version_parts = version_str.split('.')
                                                        if len(version_parts) >= 2:
                                                            capabilities["cuda_major"] = int(version_parts[0])
                                                            capabilities["cuda_minor"] = int(version_parts[1])
                                                            capabilities["cuda_version"] = version_str
                                                            self.log(f"CUDA Version from nvcc: {version_str}")
                                                            cuda_detected = True
                                                            break
                                                if cuda_detected:
                                                    break
                                except Exception as nvcc_e:
                                    self.log(f"nvcc detection failed: {nvcc_e}")
                            
                            # If still no CUDA version, infer from driver version
                            if not cuda_detected:
                                # Map common driver versions to CUDA versions
                                driver_major = int(driver_version.split('.')[0])
                                if driver_major >= 555:  # CUDA 12.x
                                    capabilities["cuda_major"] = 12
                                    capabilities["cuda_minor"] = 9
                                    capabilities["cuda_version"] = "12.9"
                                    self.log(f"Inferred CUDA 12.9 from driver {driver_version}")
                                elif driver_major >= 515:  # CUDA 11.x
                                    capabilities["cuda_major"] = 11
                                    capabilities["cuda_minor"] = 8
                                    capabilities["cuda_version"] = "11.8"
                                    self.log(f"Inferred CUDA 11.8 from driver {driver_version}")
                                    
                            return capabilities
                            
                else:
                    self.log(f"nvidia-smi returned empty or invalid output: {result.stdout}")
                    
            except subprocess.TimeoutExpired:
                self.log("nvidia-smi command timed out")
            except FileNotFoundError:
                self.log("nvidia-smi not found in PATH")
            except Exception as nvidia_e:
                self.log(f"nvidia-smi detection failed: {nvidia_e}")
        
        # If nvidia-smi worked and provided valid information
        if not cuda_detected and driver_version:
            try:
                driver_ver = float(driver_version)
                # Map driver version to CUDA compatibility
                # These mappings are based on NVIDIA compatibility matrix
                if driver_ver >= 576.0:  # RTX 50 series drivers
                    capabilities["cuda_major"] = 12
                    capabilities["cuda_minor"] = 1
                    capabilities["cuda_version"] = "12.1"
                    self.log(f"Inferred CUDA 12.1+ compatibility from driver {driver_version}")
                elif driver_ver >= 530.0:  # CUDA 12.x compatible
                    capabilities["cuda_major"] = 12
                    capabilities["cuda_minor"] = 1
                    capabilities["cuda_version"] = "12.1"
                    self.log(f"Inferred CUDA 12.1 compatibility from driver {driver_version}")
                elif driver_ver >= 450.0:  # CUDA 11.x compatible
                    capabilities["cuda_major"] = 11
                    capabilities["cuda_minor"] = 8
                    capabilities["cuda_version"] = "11.8"
                    self.log(f"Inferred CUDA 11.8 compatibility from driver {driver_version}")
                else:
                    self.log(f"Driver {driver_version} may be too old for current PyTorch")
            except ValueError:
                self.log(f"Could not parse driver version: {driver_version}")
                
            if driver_version and gpu_name:
                capabilities["nvidia_gpu"] = True
                capabilities["has_nvidia_gpu"] = True  # For compatibility
                capabilities["gpu_name"] = gpu_name
                capabilities["driver_version"] = driver_version
                self.log(f"Successfully detected NVIDIA GPU: {gpu_name}")
                self.log(f"Driver version: {driver_version}")
            else:
                self.log("No NVIDIA GPU detected")
        if not capabilities["has_nvidia_gpu"]:
            self.log("No NVIDIA GPU detected - will use CPU-only PyTorch")
        
        self.system_info = capabilities
        self.save_config()
        return capabilities

    def _is_windows(self) -> bool:
        return sys.platform == "win32"
    
    def _is_windowsapps_shim(self, p: str) -> bool:
        """Check if a path is the problematic Windows Store python shim."""
        try:
            if not p:
                return False
            p_str = str(p).lower()
            # Check for WindowsApps in path
            if "windowsapps" in p_str:
                return True
            # Also check for Microsoft Store Python locations
            if "microsoft\\windowsapps" in p_str or "local\\microsoft\\windowsapps" in p_str:
                return True
            return False
        except Exception:
            return False
    
    def _is_valid_python(self, p: str) -> bool:
        """Verify that a Python executable is real and functional."""
        try:
            if not p or not Path(p).exists():
                return False
            if self._is_windowsapps_shim(p):
                return False
            # Try to actually run it to verify it's functional
            result = subprocess.run(
                [p, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            # WindowsApps shim returns exit code 9009, real Python returns 0
            if result.returncode != 0:
                self.log(f"Python at {p} failed version check (exit {result.returncode})", "WARNING")
                return False
            return True
        except Exception as e:
            self.log(f"Failed to validate Python at {p}: {e}", "WARNING")
            return False

    def _python_info(self, p: str) -> Optional[dict]:
        """Return version and arch info for a Python executable.

        Returns dict like {"major":3, "minor":12, "arch":"64bit"} or None on failure.
        """
        try:
            code = (
                "import sys,platform,json;"
                "print(json.dumps({'major':sys.version_info.major,'minor':sys.version_info.minor,'arch':platform.architecture()[0]}))"
            )
            r = subprocess.run([p, "-c", code], capture_output=True, text=True, timeout=5)
            if r.returncode != 0:
                return None
            return json.loads(r.stdout.strip())
        except Exception:
            return None

    def _is_compatible_python(self, p: str) -> bool:
        """Require 64-bit Python 3.11+ to ensure wheel availability for NumPy/PyTorch."""
        if not self._is_valid_python(p):
            return False
        info = self._python_info(p)
        if not info:
            return False
        arch_ok = str(info.get("arch", "")).startswith("64")
        ver_ok = int(info.get("major", 0)) > 3 or (int(info.get("major", 0)) == 3 and int(info.get("minor", 0)) >= 11)
        if not arch_ok or not ver_ok:
            self.log(
                f"Incompatible Python detected: {p} (arch={info.get('arch')}, version={info.get('major')}.{info.get('minor')}). "
                f"Requiring 64-bit Python 3.11+ for compatibility.",
                "WARNING",
            )
        return arch_ok and ver_ok

    def _check_vc_redist_installed(self) -> bool:
        """Check if Microsoft Visual C++ 2015–2022 Redistributable (x64) is installed.

        Checks common registry locations for the VC runtime. Returns True if present.
        """
        if not self._is_windows():
            return True
        try:
            import winreg
            reg_paths = [
                r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
                r"SOFTWARE\Microsoft\VisualStudio\VC\Runtimes\x64",
                r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
            ]
            for root in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
                for subkey in reg_paths:
                    try:
                        key = winreg.OpenKey(root, subkey, 0, winreg.KEY_READ)
                        installed, _ = winreg.QueryValueEx(key, "Installed")
                        if int(installed) == 1:
                            self.log(f"VC++ Redistributable found at registry: {subkey}")
                            return True
                    except FileNotFoundError:
                        continue
                    except Exception as e:
                        self.log(f"VC++ registry check error at {subkey}: {e}", "WARNING")
            return False
        except Exception as e:
            self.log(f"VC++ runtime detection failed: {e}", "WARNING")
            return False

    def _install_vc_redist(self) -> bool:
        """Attempt to download and install the VC++ 2015–2022 x64 Redistributable.

        Requires internet and will prompt for elevation via UAC. Returns True on success.
        """
        if not self._is_windows():
            return True
        try:
            self.log("Attempting to install Microsoft Visual C++ Redistributable (x64)...")
            vc_url = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
            download_path = self.install_dir / "vc_redist.x64.exe"

            # Use PowerShell to download to avoid SSL issues in some environments
            ps_download = (
                f"Invoke-WebRequest -Uri '{vc_url}' -OutFile '{download_path}' -UseBasicParsing"
            )
            result = subprocess.run(["powershell", "-Command", ps_download], capture_output=True, text=True)
            if result.returncode != 0:
                self.log(f"Failed to download VC++ redistributable: {result.stderr}", "ERROR")
                return False

            # Run installer silently with elevation
            ps_install = (
                f"Start-Process -FilePath '{download_path}' -ArgumentList '/install','/quiet','/norestart' -Verb RunAs -Wait; "
                f"Write-Output $LASTEXITCODE"
            )
            result = subprocess.run(["powershell", "-Command", ps_install], capture_output=True, text=True)
            if result.returncode != 0:
                self.log(f"Failed to launch VC++ installer (UAC may have been cancelled): {result.stderr}", "WARNING")
                return False

            # Re-check installation
            if self._check_vc_redist_installed():
                self.log("VC++ Redistributable installed successfully.")
                return True
            else:
                self.log("VC++ Redistributable installation verification failed.", "WARNING")
                return False

        except Exception as e:
            self.log(f"VC++ installation step failed: {e}", "ERROR")
            return False

    def _attempt_python_install_windows(self, version: str = "3.12.7") -> Optional[str]:
        """Download and install Python per-user silently if not present.

        Returns the path to the installed python.exe if successful, else None.
        """
        if sys.platform != "win32":
            return None

        try:
            self.log("Python not found. Attempting per-user installation of Python from python.org...")
            installer_name = f"python-{version}-amd64.exe"
            download_url = f"https://www.python.org/ftp/python/{version}/{installer_name}"
            installer_path = self.install_dir / installer_name

            # Download installer via PowerShell (more reliable on Windows)
            ps_download = (
                f"Invoke-WebRequest -Uri '{download_url}' -OutFile '{installer_path}' -UseBasicParsing"
            )
            result = subprocess.run(["powershell", "-Command", ps_download], capture_output=True, text=True, timeout=180)
            if result.returncode != 0 or not installer_path.exists():
                self.log(f"Failed to download Python installer: {result.stderr}", "ERROR")
                return None

            # Silent per-user install, add to PATH, include pip
            ps_install = (
                f"Start-Process -FilePath '{installer_path}' -ArgumentList '/quiet','InstallAllUsers=0','PrependPath=1','Include_pip=1','SimpleInstall=1' -Wait; "
                f"Write-Output $LASTEXITCODE"
            )
            result = subprocess.run(["powershell", "-Command", ps_install], capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                self.log(f"Python installer returned non-zero exit. STDERR: {result.stderr}", "ERROR")
                # Do not early-return; sometimes $LASTEXITCODE is unreliable; we'll probe for python below

            # Re-probe for python using PATH, py.exe, and registry
            try:
                # Prefer py launcher to resolve the freshly installed version
                if shutil.which("py"):
                    r = subprocess.run(["py", "-3", "-c", "import sys; print(sys.executable)"], capture_output=True, text=True, timeout=10)
                    candidate = r.stdout.strip() if r.returncode == 0 else None
                    if candidate and self._is_valid_python(candidate):
                        self.log(f"Python installed at: {candidate}")
                        return candidate
            except Exception:
                pass

            cand = shutil.which("python") or shutil.which("python3")
            if cand and self._is_valid_python(cand):
                self.log(f"Python installed at (PATH): {cand}")
                return cand

            # Registry fallback
            try:
                import winreg
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"SOFTWARE\Python\PythonCore") as root:
                    i = 0
                    chosen = None
                    while True:
                        try:
                            ver = winreg.EnumKey(root, i); i += 1
                            with winreg.OpenKey(root, f"{ver}\\InstallPath") as ip:
                                try:
                                    exe, _ = winreg.QueryValueEx(ip, "ExecutablePath")
                                except FileNotFoundError:
                                    path, _ = winreg.QueryValueEx(ip, None)
                                    exe = os.path.join(path, "python.exe")
                                if exe and self._is_valid_python(exe):
                                    chosen = exe if (chosen is None or ver > chosen) else chosen
                        except OSError:
                            break
                if chosen:
                    self.log(f"Python installed at (registry): {chosen}")
                    return chosen
            except Exception:
                pass

            self.log("Python installation attempted but executable not found.", "ERROR")
            return None
        except Exception as e:
            self.log(f"Python auto-installation failed: {e}", "ERROR")
            return None

    def ensure_windows_prerequisites(self) -> bool:
        """Ensure system-level prerequisites exist (Windows only).

        - Microsoft Visual C++ 2015–2022 Redistributable (x64): REQUIRED for PyTorch DLLs
          (both CPU and CUDA builds) on Windows.
        Returns True if all checks pass or are not applicable; returns False if required
        prerequisites are missing and cannot be installed automatically.
        """
        if not self._is_windows():
            return True
        try:
            if not self._check_vc_redist_installed():
                self.log("Microsoft Visual C++ Redistributable (x64) not found.", "WARNING")
                success = self._install_vc_redist()
                if not success:
                    # Hard fail: Without VC++ runtime, torch DLLs (c10.dll) will fail to initialize even for CPU builds.
                    self.log(
                        (
                            "Required prerequisite missing: Microsoft Visual C++ 2015–2022 Redistributable (x64).\n"
                            "Unable to install it automatically (UAC cancelled, offline, or blocked).\n"
                            "Please install it manually from: https://aka.ms/vs/17/release/vc_redist.x64.exe\n"
                            "After installing, re-run the PyBattML installer."
                        ),
                        "ERROR",
                    )
                    return False
            else:
                self.log("All required Windows prerequisites found.")
            return True
        except Exception as e:
            self.log(f"Prerequisite check failed: {e}", "WARNING")
            return True
    
    def determine_pytorch_variant(self) -> Tuple[str, List[str]]:
        """Determine the best PyTorch installation command for this system"""
        
        # Debug logging
        self.log(f"DEBUG: System info keys: {list(self.system_info.keys())}")
        self.log(f"DEBUG: has_nvidia_gpu = {self.system_info.get('has_nvidia_gpu', False)}")
        self.log(f"DEBUG: nvidia_gpu = {self.system_info.get('nvidia_gpu', False)}")
        self.log(f"DEBUG: cuda_major = {self.system_info.get('cuda_major', 0)}")
        self.log(f"DEBUG: cuda_minor = {self.system_info.get('cuda_minor', 0)}")
        
        if not self.system_info.get("has_nvidia_gpu", False):
            self.log("No NVIDIA GPU detected, using CPU PyTorch")
            # Explicitly use the CPU index URL for PyTorch to avoid ambiguity
            return "cpu", ["torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"]
        
        cuda_major = self.system_info.get("cuda_major", 0)
        cuda_minor = self.system_info.get("cuda_minor", 0)
        
        # PyTorch CUDA compatibility mapping (as of 2025)
        # RTX 5070/50-series with sm_120 architecture needs cu128 for optimal performance
        if cuda_major >= 12 and cuda_minor >= 8:
            self.log(f"Using CUDA 12.8+ PyTorch (cu128) for CUDA {cuda_major}.{cuda_minor}")
            return "cu128", ["torch==2.8.0+cu128", "torchvision==0.23.0+cu128", "torchaudio==2.8.0+cu128", "--index-url", "https://download.pytorch.org/whl/cu128"]
        elif cuda_major >= 12 and cuda_minor >= 1:
            self.log(f"Using CUDA 12.1+ PyTorch for CUDA {cuda_major}.{cuda_minor}")
            return "cu121", ["torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu121"]
        elif cuda_major >= 11 and cuda_minor >= 8:
            self.log(f"Using CUDA 11.8+ PyTorch for CUDA {cuda_major}.{cuda_minor}")
            return "cu118", ["torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"]
        else:
            self.log(f"CUDA {cuda_major}.{cuda_minor} is too old for current PyTorch, using CPU version")
            return "cpu", ["torch", "torchvision", "torchaudio"]
    
    def setup_python_environment(self) -> bool:
        """Set up Python virtual environment"""
        self.log("=== Setting up Python Environment ===")
        
        try:
            # Create virtual environment
            if self.venv_dir.exists():
                self.log("Removing existing virtual environment...")
                shutil.rmtree(self.venv_dir)
            
            self.log("Creating new virtual environment...")
            
            # Get the actual Python executable (not our packaged .exe)
            if getattr(sys, 'frozen', False):
                # We're in a PyInstaller executable, locate a real system Python
                python_exe = None

                if sys.platform == "win32":
                    # 1) Prefer the Python Launcher (py.exe) – reliably points to a real install
                    try:
                        if shutil.which("py"):
                            # Force 64-bit resolution first
                            self.log("Trying to resolve Python via py.exe launcher (-3-64)...")
                            r = subprocess.run([
                                "py", "-3-64", "-c", "import sys; print(sys.executable)"
                            ], capture_output=True, text=True, timeout=10)
                            candidate = r.stdout.strip() if r.returncode == 0 else None
                            if candidate and self._is_compatible_python(candidate):
                                python_exe = candidate
                                self.log(f"Found compatible Python via py.exe (64-bit): {python_exe}")
                            else:
                                # Fallback to generic -3 and filter after
                                self.log("py -3-64 failed; trying py -3 ...", "WARNING")
                                r = subprocess.run([
                                    "py", "-3", "-c", "import sys; print(sys.executable)"
                                ], capture_output=True, text=True, timeout=10)
                                candidate = r.stdout.strip() if r.returncode == 0 else None
                                if candidate and self._is_compatible_python(candidate):
                                    python_exe = candidate
                                    self.log(f"Found compatible Python via py.exe: {python_exe}")
                    except Exception as e:
                        self.log(f"py.exe lookup failed: {e}", "WARNING")

                    # 2) Next, try PATH but avoid the Microsoft Store alias
                    if not python_exe:
                        cand = shutil.which("python") or shutil.which("python3")
                        if cand:
                            if self._is_compatible_python(cand):
                                python_exe = cand
                                self.log(f"Found compatible Python on PATH: {python_exe}")
                            else:
                                self.log(f"Skipping incompatible/WindowsApps python on PATH: {cand}", "WARNING")

                    # 3) Try Windows registry for installed Python locations
                    if not python_exe:
                        try:
                            import winreg
                            def _probe_registry(hive, base):
                                best = None
                                try:
                                    with winreg.OpenKey(hive, base) as root:
                                        i = 0
                                        while True:
                                            try:
                                                ver = winreg.EnumKey(root, i)
                                                i += 1
                                                with winreg.OpenKey(root, f"{ver}\\InstallPath") as ip:
                                                    try:
                                                        exe, _ = winreg.QueryValueEx(ip, "ExecutablePath")
                                                    except FileNotFoundError:
                                                        path, _ = winreg.QueryValueEx(ip, None)
                                                        exe = os.path.join(path, "python.exe")
                                                    if exe and self._is_valid_python(exe):
                                                        best = exe if (best is None or ver > best) else best
                                            except OSError:
                                                break
                                except FileNotFoundError:
                                    pass
                                return best

                            reg_paths = [
                                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Python\PythonCore"),
                                (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Python\PythonCore"),
                                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Python\PythonCore"),
                            ]
                            for hive, base in reg_paths:
                                cand = _probe_registry(hive, base)
                                if cand:
                                    python_exe = cand
                                    break
                        except Exception as e:
                            self.log(f"Python registry lookup failed: {e}", "WARNING")

                    # 4) Finally, check common install locations
                    if not python_exe:
                        common_paths = [
                            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python",
                            Path("C:/Program Files/Python"),
                            Path("C:/Program Files (x86)/Python"),
                            Path("C:/Python312"),
                            Path("C:/Python311"),
                            Path("C:/Python310"),
                        ]
                        for base in common_paths:
                            try:
                                if base.exists():
                                    for exe in base.rglob("python.exe"):
                                        if self._is_compatible_python(str(exe)):
                                            python_exe = str(exe)
                                            self.log(f"Found valid Python in common paths: {python_exe}")
                                            break
                            except Exception:
                                continue
                            if python_exe:
                                break
                else:
                    # Non-Windows: rely on PATH or common names
                    python_exe = shutil.which("python3") or shutil.which("python")

                if not python_exe:
                    # Try to install 64-bit Python per-user automatically
                    python_exe = self._attempt_python_install_windows()
                if not python_exe:
                    raise RuntimeError(
                        "Could not locate or install a real Python interpreter to create a virtual environment. "
                        "On Windows, avoid the Microsoft Store alias (WindowsApps). The installer attempted to "
                        "download Python from python.org but failed. Please install Python 3.11+ manually and re-run."
                    )
            else:
                python_exe = sys.executable

            # Final defensive check: never proceed with an invalid or incompatible interpreter
            if not self._is_compatible_python(python_exe) or self._is_windowsapps_shim(python_exe):
                self.log(
                    f"Selected Python appears invalid, incompatible (32-bit or <3.11), or is a WindowsApps alias: {python_exe}",
                    "WARNING",
                )
                # On Windows, try a last-chance resolution/install
                if sys.platform == "win32":
                    # Try a quick scan of common per-user install paths first
                    quick_candidates = [
                        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python" / "Python312" / "python.exe",
                        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python" / "Python311" / "python.exe",
                        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python" / "Python310" / "python.exe",
                    ]
                    for cand in quick_candidates:
                        if self._is_compatible_python(str(cand)):
                            python_exe = str(cand)
                            break
                    # If still invalid, attempt a per-user install and re-validate
                    if not self._is_compatible_python(python_exe):
                        resolved = self._attempt_python_install_windows()
                        if resolved and self._is_compatible_python(resolved):
                            python_exe = resolved
                # If still invalid at this point, abort with a clear message
                if not self._is_compatible_python(python_exe):
                    raise RuntimeError(
                        "Could not obtain a functional, 64-bit Python 3.11+ interpreter (WindowsApps alias or 32-bit was detected). "
                        "Please install Python 3.11+ x64 from python.org and disable the Microsoft Store Python alias: "
                        "Settings > Apps > Advanced app settings > App execution aliases."
                    )

            self.log(f"Using Python executable: {python_exe}")
            # Prefer invoking via launcher on Windows to avoid quoting issues
            if sys.platform == "win32" and Path(python_exe).name.lower() == "py.exe":
                subprocess.run([python_exe, "-3", "-m", "venv", str(self.venv_dir)], check=True, timeout=240)
            else:
                subprocess.run([python_exe, "-m", "venv", str(self.venv_dir)], check=True, timeout=240)
            
            # Get paths to venv executables
            if sys.platform == "win32":
                venv_python = self.venv_dir / "Scripts" / "python.exe"
                venv_pip = self.venv_dir / "Scripts" / "pip.exe"
            else:
                venv_python = self.venv_dir / "bin" / "python"
                venv_pip = self.venv_dir / "bin" / "pip"
            
            if not venv_python.exists():
                raise Exception(f"Virtual environment Python not found at {venv_python}")
            
            # Upgrade pip
            self.log("Upgrading pip...")
            subprocess.run([
                str(venv_python), "-m", "pip", "install", "--upgrade", "pip"
            ], check=True, timeout=60)
            
            self.install_config["venv_python"] = str(venv_python)
            self.install_config["venv_pip"] = str(venv_pip)
            self.save_config()
            
            return True
            
        except Exception as e:
            self.log(f"Failed to setup Python environment: {e}", "ERROR")
            return False
    
    def install_base_requirements(self) -> bool:
        """Install base requirements (non-PyTorch dependencies)"""
        self.log("=== Installing Base Requirements ===")
        
        # Core requirements without PyTorch
        base_requirements = [
            "numpy>=1.26.4",
            "pandas>=2.2.2",
            "matplotlib>=3.9.2",
            "scikit-learn>=1.5.1",
            "scipy>=1.14.1",
            "PyQt5>=5.15.11",
            "h5py>=3.11.0",
            "optuna>=3.6.1",
            "requests>=2.32.3",
            "PyYAML>=6.0.2",
            "tqdm>=4.66.5",
            "psutil>=6.0.0",
            "pillow>=10.4.0",
            "joblib>=1.4.2"
        ]
        
        try:
            venv_python = self.install_config["venv_python"]
            
            for req in base_requirements:
                self.log(f"Installing {req}...")
                subprocess.run([
                    venv_python, "-m", "pip", "install", req
                ], check=True, timeout=300)
            
            return True
            
        except Exception as e:
            self.log(f"Failed to install base requirements: {e}", "ERROR")
            return False
    
    def install_pytorch(self) -> bool:
        """Install appropriate PyTorch variant"""
        self.log("=== Installing PyTorch ===")
        
        try:
            variant, install_cmd = self.determine_pytorch_variant()
            venv_python = self.install_config["venv_python"]
            
            self.log(f"Installing PyTorch variant: {variant}")
            self.log(f"Install command: {' '.join(install_cmd)}")
            
            # Install PyTorch
            cmd = [venv_python, "-m", "pip", "install"] + install_cmd
            subprocess.run(cmd, check=True, timeout=600)  # 10 minutes timeout
            
            # Verify PyTorch installation
            verify_cmd = [
                venv_python, "-c",
                (
                    "import sys;\n"
                    "try:\n"
                    "    import torch\n"
                    "    print(f'PyTorch {torch.__version__} installed')\n"
                    "    print(f'CUDA available: {torch.cuda.is_available()}')\n"
                    "    sys.exit(0)\n"
                    "except Exception as e:\n"
                    "    import traceback\n"
                    "    print('VERIFY_IMPORT_ERROR:', e)\n"
                    "    traceback.print_exc()\n"
                    "    sys.exit(1)\n"
                )
            ]
            
            result = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=45)
            if result.returncode == 0:
                self.log("PyTorch verification successful:")
                for line in result.stdout.strip().split('\n'):
                    self.log(f"  {line}")
            else:
                self.log("PyTorch verification failed; falling back to CPU build.", "WARNING")
                self.log(f"  STDOUT: {result.stdout}", "WARNING")
                self.log(f"  STDERR: {result.stderr}", "WARNING")
                # Attempt to uninstall GPU builds and install CPU wheels
                try:
                    self.log("Uninstalling GPU PyTorch packages...")
                    subprocess.run([venv_python, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"], check=False, timeout=120)
                except Exception as ue:
                    self.log(f"Uninstall warning: {ue}", "WARNING")
                self.log("Installing CPU PyTorch packages as fallback...")
                cpu_fallback_cmd = ["torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"]
                subprocess.run([venv_python, "-m", "pip", "install"] + cpu_fallback_cmd, check=True, timeout=600)
                # Re-verify CPU install
                cpu_verify = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=45)
                if cpu_verify.returncode == 0:
                    self.log("CPU PyTorch verification successful:")
                    for line in cpu_verify.stdout.strip().split('\n'):
                        self.log(f"  {line}")
                    variant = "cpu_fallback"
                else:
                    self.log("CPU PyTorch verification still failing.", "ERROR")
                    self.log(f"  STDOUT: {cpu_verify.stdout}", "ERROR")
                    self.log(f"  STDERR: {cpu_verify.stderr}", "ERROR")
                    return False
            
            self.install_config["pytorch_variant"] = variant
            self.save_config()
            
            return True
            
        except Exception as e:
            self.log(f"Failed to install PyTorch: {e}", "ERROR")
            
            # Try to install CPU version as fallback
            try:
                self.log("Attempting CPU PyTorch as fallback...")
                cpu_fallback_cmd = ["torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"]
                fallback_cmd = [venv_python, "-m", "pip", "install"] + cpu_fallback_cmd
                subprocess.run(fallback_cmd, check=True, timeout=600)
                self.install_config["pytorch_variant"] = "cpu_fallback"
                self.save_config()
                return True
            except:
                return False
    
    def install_gpu_specific_packages(self) -> bool:
        """Install GPU-specific packages if GPU is available"""
        if not self.system_info.get("has_nvidia_gpu", False):
            return True  # Skip if no GPU
        
        self.log("=== Installing GPU-specific Packages ===")
        
        try:
            venv_python = self.install_config["venv_python"]
            
            # Install nvidia-ml-py3 for GPU monitoring
            subprocess.run([
                venv_python, "-m", "pip", "install", "nvidia-ml-py3>=7.352.0"
            ], check=True, timeout=120)
            
            return True
            
        except Exception as e:
            self.log(f"Failed to install GPU packages: {e}", "WARNING")
            return True  # Not critical, continue
    
    def create_launcher_script(self) -> bool:
        """Create launcher script that uses the virtual environment"""
        self.log("=== Creating Launcher Script ===")
        
        try:
            if sys.platform == "win32":
                launcher_path = self.install_dir / "launch_pybattml.bat"
                venv_python = self.install_config["venv_python"]
                main_script = self.install_dir / "launch_gui_qt.py"
                
                # Get venv paths for PATH
                venv_scripts_path = self.venv_dir / "Scripts"
                venv_lib_bin_path = self.venv_dir / "Library" / "bin"

                # Create launcher that sets project directory environment variable
                launcher_content = f'''@echo off
title PyBattML - Python Battery Modeling Library
echo Starting PyBattML...
echo Installation: {self.install_dir}
echo Project Data: {self.project_dir}
cd /d "{self.install_dir}"
set "PATH={venv_scripts_path};{venv_lib_bin_path};%PATH%"
set VESTIM_PROJECT_DIR={self.project_dir}
"{venv_python}" "{main_script}" %*
if errorlevel 1 (
    echo.
    echo PyBattML encountered an error.
    pause
)
'''
                
                with open(launcher_path, 'w') as f:
                    f.write(launcher_content)
                
                # Also create a project-specific launcher in the project directory
                project_launcher_path = self.project_dir / "launch_pybattml_here.bat"
                project_launcher_content = f'''@echo off
title PyBattML - Python Battery Modeling Library (Project: {self.project_dir.name})
echo Starting PyBattML with this project directory...
echo Installation: {self.install_dir}
echo Project Data: {self.project_dir}
cd /d "{self.project_dir}"
set "PATH={venv_scripts_path};{venv_lib_bin_path};%PATH%"
set VESTIM_PROJECT_DIR={self.project_dir}
"{venv_python}" "{main_script}" %*
if errorlevel 1 (
    echo.
    echo PyBattML encountered an error.
    pause
)
'''
                
                with open(project_launcher_path, 'w') as f:
                    f.write(project_launcher_content)
                
                self.install_config["launcher_script"] = str(launcher_path)
                self.install_config["project_launcher_script"] = str(project_launcher_path)
                
            else:
                # Linux/Mac launcher (similar structure)
                launcher_path = self.install_dir / "launch_vestim.sh"
                venv_python = self.install_config["venv_python"]
                main_script = self.install_dir / "launch_gui_qt.py"
                
                launcher_content = f'''#!/bin/bash
echo "Starting Vestim..."
echo "Installation: {self.install_dir}"
echo "Project Data: {self.project_dir}"
cd "{self.install_dir}"
export VESTIM_PROJECT_DIR="{self.project_dir}"
"{venv_python}" "{main_script}" "$@"
'''
                
                with open(launcher_path, 'w') as f:
                    f.write(launcher_content)
                
                # Make executable
                os.chmod(launcher_path, 0o755)
                
                # Project launcher
                project_launcher_path = self.project_dir / "launch_vestim_here.sh"
                project_launcher_content = f'''#!/bin/bash
echo "Starting Vestim with this project directory..."
echo "Installation: {self.install_dir}"
echo "Project Data: {self.project_dir}"
cd "{self.project_dir}"
export VESTIM_PROJECT_DIR="{self.project_dir}"
"{venv_python}" "{main_script}" "$@"
'''
                
                with open(project_launcher_path, 'w') as f:
                    f.write(project_launcher_content)
                
                os.chmod(project_launcher_path, 0o755)
                
                self.install_config["launcher_script"] = str(launcher_path)
                self.install_config["project_launcher_script"] = str(project_launcher_path)

            self.save_config()
            return True

        except Exception as e:
            self.log(f"Failed to create launcher script: {e}", "ERROR")
            return False

    def create_desktop_shortcut(self) -> bool:
        """Create desktop shortcut for easy access"""
        self.log("=== Creating Desktop Shortcut ===")
        
        try:
            if sys.platform == "win32":
                import os
                import subprocess
                
                desktop_path = Path(os.path.expanduser("~/Desktop"))
                shortcut_path = desktop_path / "PyBattML.lnk"
                launcher_path = self.install_config.get("launcher_script")
                
                if not launcher_path or not Path(launcher_path).exists():
                    self.log("Launcher script not found, cannot create shortcut", "ERROR")
                    return False
                
                # Get icon path
                icon_path = self.install_config.get("icon_path", "")
                
                # Create desktop shortcut using PowerShell
                ps_command = f'''
$WshShell = New-Object -comObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("{shortcut_path}")
$Shortcut.TargetPath = "{launcher_path}"
$Shortcut.WorkingDirectory = "{self.install_dir}"
$Shortcut.Description = "PyBattML - Python Battery Modeling Library"'''
                
                # Add icon if available
                if icon_path and Path(icon_path).exists():
                    ps_command += f'''
$Shortcut.IconLocation = "{icon_path}"'''
                
                ps_command += '''
$Shortcut.Save()
'''
                
                result = subprocess.run(['powershell', '-Command', ps_command], 
                                      capture_output=True, text=True)
                
                if result.returncode == 0:
                    self.log(f"Desktop shortcut created: {shortcut_path}")
                    
                    # Also create Start Menu shortcut
                    start_menu_path = Path(os.path.expanduser("~/AppData/Roaming/Microsoft/Windows/Start Menu/Programs"))
                    start_menu_shortcut = start_menu_path / "PyBattML.lnk"
                    
                    ps_command_start = f'''
$WshShell = New-Object -comObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("{start_menu_shortcut}")
$Shortcut.TargetPath = "{launcher_path}"
$Shortcut.WorkingDirectory = "{self.install_dir}"
$Shortcut.Description = "PyBattML - Python Battery Modeling Library"'''
                    
                    # Add icon if available
                    if icon_path and Path(icon_path).exists():
                        ps_command_start += f'''
$Shortcut.IconLocation = "{icon_path}"'''
                    
                    ps_command_start += '''
$Shortcut.Save()
'''
                    
                    result_start = subprocess.run(['powershell', '-Command', ps_command_start], 
                                                capture_output=True, text=True)
                    
                    if result_start.returncode == 0:
                        self.log(f"Start Menu shortcut created: {start_menu_shortcut}")
                    else:
                        self.log(f"Start Menu shortcut creation failed: {result_start.stderr}", "WARNING")
                    
                    # Also create an Uninstall shortcut in Start Menu
                    uninstaller_path = self.install_config.get("uninstaller_script")
                    if uninstaller_path and Path(uninstaller_path).exists():
                        uninstall_shortcut = start_menu_path / "Uninstall PyBattML.lnk"
                        
                        ps_command_uninstall = f'''
$WshShell = New-Object -comObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("{uninstall_shortcut}")
$Shortcut.TargetPath = "{uninstaller_path}"
$Shortcut.WorkingDirectory = "{self.install_dir}"
$Shortcut.Description = "Uninstall PyBattML"'''
                        
                        if icon_path and Path(icon_path).exists():
                            ps_command_uninstall += f'''
$Shortcut.IconLocation = "{icon_path}"'''
                        
                        ps_command_uninstall += '''
$Shortcut.Save()
'''
                        
                        result_uninstall = subprocess.run(['powershell', '-Command', ps_command_uninstall], 
                                                    capture_output=True, text=True)
                        
                        if result_uninstall.returncode == 0:
                            self.log(f"Uninstall shortcut created in Start Menu: {uninstall_shortcut}")
                        else:
                            self.log(f"Uninstall shortcut creation failed: {result_uninstall.stderr}", "WARNING")
                        
                    return True
                else:
                    self.log(f"Desktop shortcut creation failed: {result.stderr}", "ERROR")
                    return False
            else:
                # Linux/Mac desktop shortcut creation would go here
                self.log("Desktop shortcut creation not implemented for this platform")
                return True
                
        except Exception as e:
            self.log(f"Failed to create desktop shortcut: {e}", "ERROR")
            return False

    def cleanup_old_start_menu_entries(self) -> bool:
        """Clean up old PyBattML Start Menu entries"""
        self.log("=== Cleaning Up Old Start Menu Entries ===")
        
        try:
            if sys.platform == "win32":
                import os
                import shutil
                
                # Check both user and common start menu locations
                start_menu_locations = [
                    Path(os.path.expanduser("~/AppData/Roaming/Microsoft/Windows/Start Menu/Programs")),
                    Path("C:/ProgramData/Microsoft/Windows/Start Menu/Programs")
                ]
                
                cleaned_count = 0
                for start_menu_path in start_menu_locations:
                    if start_menu_path.exists():
                        # Look for PyBattML-related items
                        for item in start_menu_path.rglob("*PyBattML*"):
                            try:
                                if item.is_file():
                                    item.unlink()
                                    cleaned_count += 1
                                    self.log(f"Removed old start menu file: {item}")
                                elif item.is_dir():
                                    shutil.rmtree(item)
                                    cleaned_count += 1
                                    self.log(f"Removed old start menu folder: {item}")
                            except Exception as e:
                                self.log(f"Could not remove {item}: {e}", "WARNING")
                
                self.log(f"Cleaned up {cleaned_count} old Start Menu entries")
                return True
            else:
                self.log("Start Menu cleanup not implemented for this platform")
                return True
                
        except Exception as e:
            self.log(f"Failed to clean up old Start Menu entries: {e}", "WARNING")
            return True  # Don't fail installation for this

    def create_uninstaller(self) -> bool:
        """Create uninstaller script and registry entry"""
        self.log("=== Creating Uninstaller ===")
        
        try:
            if sys.platform == "win32":
                import subprocess
                
                # Create uninstaller script
                uninstaller_path = self.install_dir / "uninstall_pybattml.bat"
                
                uninstaller_content = f'''@echo off
title Uninstall PyBattML - Python Battery Modeling Library
echo.
echo This will completely remove PyBattML and all its components.
echo Installation directory: {self.install_dir}
echo Project directory: {self.project_dir}
echo.
set /p confirm="Are you sure you want to uninstall PyBattML? (Y/N): "
if /i "%confirm%" neq "Y" (
    echo Uninstall cancelled.
    pause
    exit /b 0
)

echo.
echo Uninstalling PyBattML...

:: Remove desktop shortcut
echo Removing desktop shortcuts...
if exist "%USERPROFILE%\\Desktop\\PyBattML.lnk" del "%USERPROFILE%\\Desktop\\PyBattML.lnk"

:: Remove Start Menu shortcuts
echo Removing Start Menu shortcuts...
if exist "%APPDATA%\\Microsoft\\Windows\\Start Menu\\Programs\\PyBattML.lnk" del "%APPDATA%\\Microsoft\\Windows\\Start Menu\\Programs\\PyBattML.lnk"

:: Remove registry entries
echo Removing registry entries...
reg delete "HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\PyBattML" /f >nul 2>&1
reg delete "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\PyBattML" /f >nul 2>&1

:: Stop any running processes
echo Stopping PyBattML processes...
taskkill /f /im python.exe /fi "WINDOWTITLE eq *PyBattML*" >nul 2>&1

:: Remove installation directory
echo Removing installation directory...
cd /d "%TEMP%"
rmdir /s /q "{self.install_dir}" >nul 2>&1

:: Ask about project directory
echo.
set /p remove_project="Remove project directory and data? (Y/N): "
if /i "%remove_project%" equ "Y" (
    echo Removing project directory...
    rmdir /s /q "{self.project_dir}" >nul 2>&1
)

echo.
echo PyBattML has been uninstalled successfully.
echo Thank you for using PyBattML!
pause
'''
                
                with open(uninstaller_path, 'w') as f:
                    f.write(uninstaller_content)
                
                # Create registry entry for Programs & Features
                app_name = "PyBattML - Python Battery Modeling Library"
                version = "2.0.0"
                
                # Calculate estimated size (in KB)
                estimated_size = 0
                try:
                    for root, dirs, files in os.walk(self.install_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            if os.path.exists(file_path):
                                estimated_size += os.path.getsize(file_path)
                    estimated_size = estimated_size // 1024  # Convert to KB
                except:
                    estimated_size = 500000  # Default 500MB estimate
                
                # Registry command - try HKLM first, fallback to HKCU
                reg_commands_hklm = f'''
reg add "HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\PyBattML" /v "DisplayName" /t REG_SZ /d "{app_name}" /f
reg add "HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\PyBattML" /v "DisplayVersion" /t REG_SZ /d "{version}" /f
reg add "HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\PyBattML" /v "Publisher" /t REG_SZ /d "PyBattML Team" /f
reg add "HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\PyBattML" /v "UninstallString" /t REG_SZ /d "\\"{uninstaller_path}\\"" /f
reg add "HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\PyBattML" /v "InstallLocation" /t REG_SZ /d "{self.install_dir}" /f
reg add "HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\PyBattML" /v "EstimatedSize" /t REG_DWORD /d {estimated_size} /f
reg add "HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\PyBattML" /v "NoModify" /t REG_DWORD /d 1 /f
reg add "HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\PyBattML" /v "NoRepair" /t REG_DWORD /d 1 /f
'''
                
                reg_commands_hkcu = f'''
reg add "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\PyBattML" /v "DisplayName" /t REG_SZ /d "{app_name}" /f
reg add "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\PyBattML" /v "DisplayVersion" /t REG_SZ /d "{version}" /f
reg add "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\PyBattML" /v "Publisher" /t REG_SZ /d "PyBattML Team" /f
reg add "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\PyBattML" /v "UninstallString" /t REG_SZ /d "\\"{uninstaller_path}\\"" /f
reg add "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\PyBattML" /v "InstallLocation" /t REG_SZ /d "{self.install_dir}" /f
reg add "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\PyBattML" /v "EstimatedSize" /t REG_DWORD /d {estimated_size} /f
reg add "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\PyBattML" /v "NoModify" /t REG_DWORD /d 1 /f
reg add "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\PyBattML" /v "NoRepair" /t REG_DWORD /d 1 /f
'''
                
                # Try to add registry entries (try HKLM first, fallback to HKCU)
                try:
                    # Try system-wide registration first (requires admin)
                    result = subprocess.run(['powershell', '-Command', reg_commands_hklm], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        self.log("System-wide uninstall registry entry created successfully")
                    else:
                        # Fallback to user-specific registration
                        self.log("System-wide registration failed, trying user-specific...", "WARNING")
                        result = subprocess.run(['powershell', '-Command', reg_commands_hkcu], 
                                              capture_output=True, text=True)
                        if result.returncode == 0:
                            self.log("User-specific uninstall registry entry created successfully")
                        else:
                            self.log(f"Both registry entry attempts failed: {result.stderr}", "WARNING")
                except Exception as e:
                    self.log(f"Registry entry creation failed: {e}", "WARNING")
                
                self.install_config["uninstaller_script"] = str(uninstaller_path)
                self.log(f"Uninstaller created: {uninstaller_path}")
                return True
            else:
                self.log("Uninstaller creation not implemented for this platform")
                return True
                
        except Exception as e:
            self.log(f"Failed to create uninstaller: {e}", "ERROR")
            return False

    def save_config(self):
        """Save installation configuration"""
        config = {
            "system_info": self.system_info,
            "install_config": self.install_config,
            "setup_completed": self.is_setup_complete(),
            "setup_timestamp": time.time()
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_config(self) -> bool:
        """Load existing installation configuration"""
        if not self.config_file.exists():
            return False
        
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            self.system_info = config.get("system_info", {})
            self.install_config = config.get("install_config", {})
            return True
            
        except Exception as e:
            self.log(f"Failed to load config: {e}", "WARNING")
            return False
    
    def is_setup_complete(self) -> bool:
        """Check if environment setup is complete and valid"""
        if not self.venv_dir.exists():
            return False
        
        venv_python = self.install_config.get("venv_python")
        if not venv_python or not Path(venv_python).exists():
            return False
        
        launcher = self.install_config.get("launcher_script")
        if not launcher or not Path(launcher).exists():
            return False
        
        return True
    
    def extract_application_files(self) -> bool:
        """Extract Vestim application files from executable"""
        self.log("=== Extracting Application Files ===")
        
        try:
            import sys
            
            # Check if running from PyInstaller bundle
            if hasattr(sys, '_MEIPASS'):
                # PyInstaller bundle - extract from temp directory
                bundle_dir = Path(sys._MEIPASS)
                
                # INSTALLATION DIRECTORY: Core Vestim modules (for running the app)
                vestim_source = bundle_dir / "vestim"
                vestim_dest = self.install_dir / "vestim"
                
                if vestim_source.exists():
                    if vestim_dest.exists():
                        shutil.rmtree(vestim_dest)
                    shutil.copytree(vestim_source, vestim_dest)
                    self.log(f"Extracted Vestim modules to installation: {vestim_dest}")
                else:
                    self.log("Warning: 'vestim' package not found in installer bundle; application may fail to launch.", "WARNING")
                
                # Copy launch_gui_qt.py to installation directory
                launch_script_source = bundle_dir / "launch_gui_qt.py"
                launch_script_dest = self.install_dir / "launch_gui_qt.py"
                
                if launch_script_source.exists():
                    shutil.copy2(launch_script_source, launch_script_dest)
                    self.log(f"Copied launch script to installation: {launch_script_dest}")
                else:
                    self.log("Warning: launch_gui_qt.py not found in bundle", "WARNING")
                
                # Copy standalone test CSV file to installation directory
                csv_source = bundle_dir / "119_ReorderedUS06_n20C.csv"
                csv_dest = self.install_dir / "119_ReorderedUS06_n20C.csv"
                
                if csv_source.exists():
                    shutil.copy2(csv_source, csv_dest)
                    self.log(f"Copied test CSV to installation: {csv_dest}")
                
                # Copy USER_README.md to installation directory
                readme_source = bundle_dir / "USER_README.md"
                readme_dest = self.install_dir / "USER_README.md"
                
                if readme_source.exists():
                    shutil.copy2(readme_source, readme_dest)
                    self.log(f"Copied user readme to installation: {readme_dest}")
                
                # Copy PyBattML icon to installation directory for shortcuts
                icon_source = bundle_dir / "vestim" / "gui" / "resources" / "PyBattML_icon.ico"
                icon_dest = self.install_dir / "PyBattML_icon.ico"
                
                if icon_source.exists():
                    shutil.copy2(icon_source, icon_dest)
                    self.log(f"Copied icon to installation: {icon_dest}")
                    self.install_config["icon_path"] = str(icon_dest)
                else:
                    self.log("Warning: PyBattML icon not found in bundle", "WARNING")
                
                # PROJECT DIRECTORY: User data and templates (separate location)
                self.project_dir.mkdir(parents=True, exist_ok=True)
                
                # Extract defaults_templates to project directory
                templates_source = bundle_dir / "defaults_templates"
                templates_dest = self.project_dir / "defaults_templates"
                
                if templates_source.exists():
                    if templates_dest.exists():
                        shutil.rmtree(templates_dest)
                    shutil.copytree(templates_source, templates_dest)
                    self.log(f"Extracted templates to project: {templates_dest}")
                else:
                    self.log("Warning: 'defaults_templates' not found in installer bundle; proceeding without copying.", "WARNING")
                
                # Extract data directory to project directory
                data_source = bundle_dir / "data"
                data_dest = self.project_dir / "data"

                if data_source.exists():
                    if data_dest.exists():
                        shutil.rmtree(data_dest)
                    shutil.copytree(data_source, data_dest)
                    self.log(f"Extracted data to project: {data_dest}")

                    # Verify required subfolders exist and contain at least one file; if not, create empty structure
                    def _has_files(p: Path) -> bool:
                        try:
                            return any(p.rglob("*.*"))
                        except Exception:
                            return False
                    required = [data_dest / "train_data", data_dest / "val_data", data_dest / "test_data"]
                    if not all(d.exists() for d in required) or not any(_has_files(d) for d in required):
                        self.log("Data directory missing required content; creating empty train/val/test folders.", "WARNING")
                        for sub in ("train_data", "val_data", "test_data"):
                            (data_dest / sub).mkdir(parents=True, exist_ok=True)
                else:
                    self.log("Warning: 'data' directory not found in installer bundle; creating empty train/val/test folders.", "WARNING")
                    for sub in ("train_data", "val_data", "test_data"):
                        (data_dest / sub).mkdir(parents=True, exist_ok=True)
                
                # Save project directory location
                self.install_config["project_dir"] = str(self.project_dir)
                
                # Initialize default settings to use the extracted data directories
                self.initialize_default_settings()
                
                # Create configuration file for ConfigManager
                self.create_config_file()
                
            else:
                self.log("Not running from PyInstaller - skipping extraction")
            
            return True
            
        except Exception as e:
            self.log(f"Failed to extract application files: {e}", "ERROR")
            return False

    def initialize_default_settings(self) -> bool:
        """Initialize default settings to point to the extracted data directories"""
        self.log("=== Initializing Default Settings ===")
        
        try:
            import json
            from pathlib import Path
            
            # Define the default data directories
            data_dir = self.project_dir / "data"
            default_folders = {
                "train_folder": str(data_dir / "train_data"),
                "val_folder": str(data_dir / "val_data"), 
                "test_folder": str(data_dir / "test_data"),
                "file_format": "csv"
            }
            
            # Create the default settings structure
            default_settings = {
                "last_used": default_folders.copy(),
                "default_folders": default_folders.copy()
            }
            
            # Save to the project directory so ConfigManager can find it
            settings_file = self.project_dir / "default_settings.json"
            
            with open(settings_file, 'w') as f:
                json.dump(default_settings, f, indent=2)
            
            self.log(f"Initialized default settings: {settings_file}")
            self.log(f"  - Training data: {default_folders['train_folder']}")
            self.log(f"  - Validation data: {default_folders['val_folder']}")
            self.log(f"  - Test data: {default_folders['test_folder']}")
            
            return True
            
        except Exception as e:
            self.log(f"Failed to initialize default settings: {e}", "ERROR")
            return False

    def create_config_file(self) -> bool:
        """Create vestim_config.json file for ConfigManager to read"""
        self.log("=== Creating Configuration File ===")
        
        try:
            import json
            
            # Create configuration for the installed application
            install_date = time.strftime("%Y-%m-%d %H:%M:%S")
            config = {
                "projects_directory": str(self.project_dir),
                "data_directory": str(self.project_dir / "data"),
                "defaults_directory": str(self.project_dir / "defaults_templates"),
                "install_directory": str(self.install_dir),
                "install_date": install_date,
                "version": "2.0.1"
            }
            
            # Save config file to installation directory so ConfigManager can find it
            config_file = self.install_dir / "vestim_config.json"
            
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.log(f"Created configuration file: {config_file}")
            self.log(f"  - Projects directory: {config['projects_directory']}")
            self.log(f"  - Data directory: {config['data_directory']}")
            
            return True
            
        except Exception as e:
            self.log(f"Failed to create configuration file: {e}", "ERROR")
            return False

    def run_full_setup(self) -> bool:
        """Run complete environment setup process"""
        try:
            self.log("=== Starting PyBattML Environment Setup ===")
            
            # Extract application files first
            if not self.extract_application_files():
                self.log("Failed to extract application files, but continuing...", "WARNING")
            
            # Detect system capabilities
            self.system_info = self.detect_system_capabilities()
            
            # Setup Python environment
            if not self.setup_python_environment():
                return False

            # Ensure Windows prerequisites (VC++ runtime) before heavy installs
            self.ensure_windows_prerequisites()
            
            # Install base requirements
            if not self.install_base_requirements():
                return False
            
            # Install PyTorch
            if not self.install_pytorch():
                return False
            
            # Install GPU packages
            if not self.install_gpu_specific_packages():
                self.log("GPU package installation failed, but continuing...", "WARNING")
            
            # Create launcher
            if not self.create_launcher_script():
                return False
                
            # Clean up old Start Menu entries
            self.cleanup_old_start_menu_entries()
            
            # Create uninstaller (must be before desktop shortcut so uninstall shortcut can be added)
            if not self.create_uninstaller():
                self.log("Uninstaller creation failed, but continuing...", "WARNING")
                
            # Create desktop shortcut (includes uninstall shortcut in Start Menu)
            if not self.create_desktop_shortcut():
                self.log("Desktop shortcut creation failed, but continuing...", "WARNING")
            
            self.log("=== Environment Setup Complete ===")
            return True
        finally:
            # Always cleanup the flag
            SmartEnvironmentSetup._SETUP_IN_PROGRESS = False


def main():
    """Main entry point for standalone execution"""
    if len(sys.argv) < 2:
        print("Usage: smart_environment_setup.py <install_directory> [project_directory]")
        print("  install_directory: Where Vestim application and environment will be installed")
        print("  project_directory: Where user data and templates will be placed (optional)")
        sys.exit(1)
    
    install_dir = sys.argv[1]
    project_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    def console_log(message):
        print(message)
        sys.stdout.flush()
    
    setup = SmartEnvironmentSetup(install_dir, project_dir, console_log)
    
    # Check if already set up
    if setup.load_config() and setup.is_setup_complete():
        print("Environment already set up and valid.")
        print(f"Installation directory: {setup.install_dir}")
        print(f"Project directory: {setup.project_dir}")
        sys.exit(0)
    
    # Run setup
    success = setup.run_full_setup()
    if success:
        print(f"\n=== Setup Complete ===")
        print(f"Installation directory: {setup.install_dir}")
        print(f"Project directory: {setup.project_dir}")
        print(f"Main launcher: {setup.install_config.get('launcher_script', 'N/A')}")
        print(f"Project launcher: {setup.install_config.get('project_launcher_script', 'N/A')}")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()