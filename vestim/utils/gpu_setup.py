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
    """Install the GPU-enabled version of PyTorch."""
    print("Attempting to install PyTorch with CUDA support...")
    try:
        # Command to install PyTorch with CUDA support
        # This command might need to be adjusted based on the latest PyTorch installation instructions
        command = [
            sys.executable, "-m", "pip", "install", 
            "torch==2.4.0", "--index-url", "https://download.pytorch.org/whl/cu118"
        ]
        
        # Execute the command
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Print output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        stderr = process.communicate()[1]
        if process.returncode != 0:
            print("Error installing PyTorch with CUDA support:")
            print(stderr)
            return False
        else:
            print("PyTorch with CUDA support installed successfully.")
            return True
            
    except Exception as e:
        print(f"An error occurred during installation: {e}")
        return False

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
