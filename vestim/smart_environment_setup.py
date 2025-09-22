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
            return "cpu", ["torch", "torchvision", "torchaudio"]
        
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
                # We're in a PyInstaller executable, find Python
                python_exe = shutil.which("python") or shutil.which("python3")
                if not python_exe:
                    # Try common Python locations
                    for py_path in [
                        Path(sys.prefix) / "python.exe",
                        Path("C:\\Python312\\python.exe"),
                        Path("C:\\Python311\\python.exe"),
                        Path("C:\\Python310\\python.exe")
                    ]:
                        if py_path.exists():
                            python_exe = str(py_path)
                            break
                if not python_exe:
                    raise RuntimeError("Could not find Python executable for virtual environment creation")
            else:
                python_exe = sys.executable
                
            self.log(f"Using Python executable: {python_exe}")
            subprocess.run([
                python_exe, "-m", "venv", str(self.venv_dir)
            ], check=True, timeout=120)
            
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
                "import torch; print(f'PyTorch {torch.__version__} installed'); print(f'CUDA available: {torch.cuda.is_available()}')"
            ]
            
            result = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                self.log("PyTorch verification successful:")
                for line in result.stdout.strip().split('\n'):
                    self.log(f"  {line}")
            else:
                self.log(f"PyTorch verification failed: {result.stderr}", "WARNING")
            
            self.install_config["pytorch_variant"] = variant
            self.save_config()
            
            return True
            
        except Exception as e:
            self.log(f"Failed to install PyTorch: {e}", "ERROR")
            
            # Try to install CPU version as fallback
            try:
                self.log("Attempting CPU PyTorch as fallback...")
                fallback_cmd = [venv_python, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]
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
                
                # Create launcher that sets project directory environment variable
                launcher_content = f'''@echo off
title PyBattML - Python Battery Modeling Library
echo Starting PyBattML...
echo Installation: {self.install_dir}
echo Project Data: {self.project_dir}
cd /d "{self.install_dir}"
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
                
                # Create desktop shortcut using PowerShell
                ps_command = f'''
$WshShell = New-Object -comObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("{shortcut_path}")
$Shortcut.TargetPath = "{launcher_path}"
$Shortcut.WorkingDirectory = "{self.install_dir}"
$Shortcut.Description = "PyBattML - Python Battery Modeling Library"
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
$Shortcut.Description = "PyBattML - Python Battery Modeling Library"
$Shortcut.Save()
'''
                    
                    result_start = subprocess.run(['powershell', '-Command', ps_command_start], 
                                                capture_output=True, text=True)
                    
                    if result_start.returncode == 0:
                        self.log(f"Start Menu shortcut created: {start_menu_shortcut}")
                    else:
                        self.log(f"Start Menu shortcut creation failed: {result_start.stderr}", "WARNING")
                        
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
                
                # Extract data directory to project directory
                data_source = bundle_dir / "data"
                data_dest = self.project_dir / "data"
                
                if data_source.exists():
                    if data_dest.exists():
                        shutil.rmtree(data_dest)
                    shutil.copytree(data_source, data_dest)
                    self.log(f"Extracted data to project: {data_dest}")
                
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
            config = {
                "projects_directory": str(self.project_dir),
                "data_directory": str(self.project_dir / "data"),
                "defaults_directory": str(self.project_dir / "defaults_templates"),
                "install_directory": str(self.install_dir),
                "install_date": self.log_timestamp,
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
                
            # Create desktop shortcut
            if not self.create_desktop_shortcut():
                self.log("Desktop shortcut creation failed, but continuing...", "WARNING")
                
            # Create uninstaller
            if not self.create_uninstaller():
                self.log("Uninstaller creation failed, but continuing...", "WARNING")
            
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