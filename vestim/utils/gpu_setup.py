import subprocess
import sys
import torch

def check_gpu():
    """Check if a compatible NVIDIA GPU is available."""
    try:
        # Check if CUDA is available to PyTorch
        if torch.cuda.is_available():
            print("NVIDIA GPU with CUDA support detected.")
            return True
        else:
            print("No compatible NVIDIA GPU found by PyTorch.")
            return False
    except Exception as e:
        print(f"An error occurred while checking for GPU: {e}")
        return False

def install_gpu_pytorch():
    """
    Install the GPU-enabled version of PyTorch.
    Returns a tuple (success, logs).
    """
    print("Attempting to install PyTorch with CUDA support...")
    logs = ""
    try:
        command = [
            sys.executable, "-m", "pip", "install",
            "torch==2.4.0", "--index-url", "https://download.pytorch.org/whl/cu118"
        ]
        
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False  # Do not raise exception on non-zero exit code
        )
        
        logs += "--- Installation Logs ---\n"
        logs += f"Command: {' '.join(command)}\n"
        logs += f"Exit Code: {result.returncode}\n"
        logs += "--- STDOUT ---\n"
        logs += result.stdout + "\n"
        logs += "--- STDERR ---\n"
        logs += result.stderr + "\n"

        if result.returncode == 0 and "Successfully installed" in result.stdout:
            print("PyTorch with CUDA support installed successfully.")
            return True, logs
        else:
            print("Error installing PyTorch with CUDA support:")
            print(logs)
            return False, logs
            
    except Exception as e:
        error_message = f"An unexpected error occurred during installation: {e}"
        print(error_message)
        logs += error_message
        return False, logs

def main():
    """Main function to handle GPU setup."""
    print("Checking for GPU and PyTorch installation...")
    
    if not check_gpu():
        print("The current PyTorch installation does not have CUDA support.")
        
        # Here you would typically ask the user for confirmation
        # For this example, we'll just proceed with the installation attempt
        
        if install_gpu_pytorch():
            print("GPU setup complete. Please restart the application.")
        else:
            print("GPU setup failed. The application will continue to run on the CPU.")
    else:
        print("PyTorch with CUDA support is already installed and configured.")

if __name__ == "__main__":
    main()
