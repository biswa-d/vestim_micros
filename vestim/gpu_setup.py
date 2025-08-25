import subprocess
import sys
import os
import tempfile

def install(package, log_file):
    """Install a package using pip and log the output"""
    command = f'"{sys.executable}" -m pip install {package}'
    log_file.write(f"Executing: {command}\n")
    try:
        # Using shell=True to handle complex commands with arguments like --index-url
        subprocess.check_call(command, shell=True, stdout=log_file, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        log_file.write(f"\n\nERROR: Failed to install {package}. Process returned error code {e.returncode}\n")
        # No need to write stdout/stderr as it's already redirected to the log file
        raise  # Re-raise the exception to stop the installation

def get_cuda_version(log_file):
    """Run nvidia-smi to get the CUDA version."""
    try:
        # nvidia-smi is usually in the PATH if drivers are installed.
        result = subprocess.check_output("nvidia-smi --query-gpu=cuda_version --format=csv,noheader", shell=True)
        version_str = result.decode('utf-8').strip()
        log_file.write(f"nvidia-smi reported CUDA version: {version_str}\n")
        major, minor = map(int, version_str.split('.'))
        return (major, minor)
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        log_file.write(f"Could not determine CUDA version from nvidia-smi. Error: {e}\n")
        return None

def install_torch(log_file):
    """Install the correct version of PyTorch based on the detected CUDA version."""
    log_file.write("--- Determining correct PyTorch version ---\n")
    cuda_version = get_cuda_version(log_file)
    
    install_command = "torch torchvision torchaudio" # Default to CPU
    
    if cuda_version:
        # Map detected CUDA version to available PyTorch wheels
        # This logic should be updated as PyTorch releases new wheels
        if cuda_version >= (12, 1):
            # For CUDA 12.1 and newer, the cu121 wheel is compatible.
            log_file.write("CUDA version >= 12.1 detected. Selecting PyTorch with cu121.\n")
            install_command += " --index-url https://download.pytorch.org/whl/cu121"
        elif cuda_version >= (11, 8):
            # For CUDA 11.8, use the cu118 wheel.
            log_file.write("CUDA version >= 11.8 detected. Selecting PyTorch with cu118.\n")
            install_command += " --index-url https://download.pytorch.org/whl/cu118"
        else:
            log_file.write("Detected CUDA version is older than 11.8. Falling back to CPU-only PyTorch.\n")
    else:
        log_file.write("No NVIDIA GPU/driver detected, or nvidia-smi failed. Installing CPU-only PyTorch.\n")

    log_file.write("\n--- Installing PyTorch ---\n")
    try:
        install(install_command, log_file)
    finally:
        log_file.write("--- PyTorch installation attempt finished ---\n\n")


def install_requirements(log_file):
    """Install the other required packages"""
    log_file.write("--- Installing other requirements ---\n")
    requirements_file = "requirements_cpu.txt"
    if not os.path.exists(requirements_file):
        log_file.write(f"ERROR: {requirements_file} not found!\n")
        return

    with open(requirements_file, "r") as f:
        for line in f:
            package = line.strip()
            if package and not package.startswith('#'):
                log_file.write(f"Installing {package}...\n")
                install(package, log_file)
    log_file.write("--- Finished installing other requirements ---\n")


if __name__ == "__main__":
    log_dir = tempfile.gettempdir()
    log_path = os.path.join(log_dir, "vestim_installer_log.txt")
    
    try:
        with open(log_path, "w") as log_file:
            log_file.write("Starting Vestim dependency installation.\n")
            log_file.write(f"Python executable: {sys.executable}\n\n")
            
            install_torch(log_file)
            install_requirements(log_file)
            
            log_file.write("\nInstallation completed successfully.\n")
            
    except Exception as e:
        with open(log_path, "a") as log_file:
            log_file.write(f"\n\nAN ERROR OCCURRED: {e}\n")
        # Exit with a non-zero code to signal failure to the installer
        sys.exit(1)
