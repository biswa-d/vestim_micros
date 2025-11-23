import sys
import os

# --- Start of Environment Sanity Check ---
# This block MUST run before any other imports (especially PyTorch)
# It ensures that the Python interpreter can find all necessary DLLs
# when running from a bundled installer on a clean system.

def _configure_dll_search_paths():
    """
    Dynamically adds essential DLL search paths for the application's
    virtual environment. This is critical for allowing PyTorch's C++
    extensions to find their dependencies on Windows.
    """
    if sys.platform != "win32":
        return  # This logic is specific to Windows

    # Check if we are running inside a virtual environment
    venv_path = os.environ.get("VIRTUAL_ENV")
    if not venv_path and hasattr(sys, 'prefix') and sys.prefix != sys.base_prefix:
        venv_path = sys.prefix
        
    if venv_path:
        # These are the standard locations for DLLs in a venv
        dll_paths_to_add = [
            os.path.join(venv_path, "Scripts"),
            os.path.join(venv_path, "Library", "bin")
        ]
        
        for path in dll_paths_to_add:
            if os.path.isdir(path):
                try:
                    # os.add_dll_directory is the recommended way for Python 3.8+
                    os.add_dll_directory(path)
                except AttributeError:
                    # For older Python, fall back to modifying PATH
                    os.environ["PATH"] = f"{path};{os.environ['PATH']}"

# Execute the configuration immediately
_configure_dll_search_paths()
# --- End of Environment Sanity Check ---


import multiprocessing
import signal
import atexit
import platform
import shutil

def _add_torch_lib_to_dll_path():
    """Add torch/lib directory to DLL search path proactively on Windows.

    This helps when the system DLL search order fails to locate c10.dll dependencies.
    """
    if sys.platform != "win32":
        return
    try:
        # Typical torch lib path in a venv
        torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
        if os.path.isdir(torch_lib):
            try:
                os.add_dll_directory(torch_lib)
            except AttributeError:
                # Pre-Python 3.8 fallback
                os.environ["PATH"] = f"{torch_lib};{os.environ['PATH']}"
    except Exception:
        # Best-effort only
        pass

def _preflight_check_torch():
    """Attempt to import torch early and provide actionable diagnostics on failure."""
    try:
        import importlib
        importlib.invalidate_caches()
        torch = importlib.import_module("torch")
        # Optional: print minimal info to logs/stdout for debugging (main process only)
        try:
            import multiprocessing as _mp
            is_main = (_mp.current_process().name == 'MainProcess')
        except Exception:
            is_main = True
        if is_main:
            # CRITICAL: Do NOT call torch.cuda.is_available() in the main process!
            # On Windows with multiprocessing spawn, calling torch.cuda.is_available()
            # initializes the CUDA context in the parent process, which corrupts the
            # CUDA state in child processes. This causes "operation failed due to a
            # previous error during capture" when using CUDA Graphs in subprocesses.
            # Check CUDA built status only (doesn't initialize CUDA context)
            cuda_built = getattr(getattr(torch, 'backends', None), 'cuda', None) and torch.backends.cuda.is_built()
            print(f"Detected PyTorch {getattr(torch, '__version__', 'unknown')} | CUDA built: {cuda_built} | CUDA avail: (checked in subprocess)")
        return True
    except OSError as e:
        if sys.platform == "win32" and ("WinError 1114" in str(e) or "c10.dll" in str(e)):
            msg_lines = [
                "PyTorch failed to initialize its DLLs (c10.dll).",
                "This is most often caused by a missing Microsoft Visual C++ Redistributable (x64) or an incompatible PyTorch build.",
                "",
                "How to fix:",
                "  1) Install 'Microsoft Visual C++ 2015–2022 Redistributable (x64)':",
                "     https://aka.ms/vs/17/release/vc_redist.x64.exe",
                "  2) Ensure a CPU-only PyTorch is installed in this app's venv:",
                "     - Uninstall any existing torch/torchvision/torchaudio",
                "     - Reinstall with the CPU index URL:",
                "       pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
                "  3) Verify with: python -c \"import torch; print(torch.__version__, torch.version.cuda)\"",
                "  4) Then relaunch PyBattML.",
            ]
            print("\n" + "\n".join(msg_lines))
            try:
                input("\nPress Enter to exit...")
            except Exception:
                pass
            sys.exit(1)
        else:
            raise

# Ensure torch DLLs can be found before any heavy imports
_add_torch_lib_to_dll_path()
_preflight_check_torch()

from PyQt5.QtWidgets import QApplication
from vestim.gui.src.welcome_gui_qt import WelcomeGUI
from vestim.utils import gpu_setup
from vestim.config_manager import get_config_manager
import logging

# Set up a logger for the launcher
logger = logging.getLogger(__name__)

# Global variables to track processes and applications
_active_dataloaders = []
_qt_application = None

def cleanup_dataloader_processes():
    """Clean up any remaining DataLoader worker processes."""
    try:
        if platform.system() != 'Linux':
            return  # Windows handles this automatically
            
        import psutil
        current_process = psutil.Process()
        
        # Find and terminate DataLoader worker processes
        terminated_count = 0
        for child in current_process.children(recursive=True):
            try:
                cmdline = child.cmdline()
                if cmdline and any('python' in str(arg).lower() for arg in cmdline):
                    # This could be a DataLoader worker - terminate it
                    logger.debug(f"Terminating potential DataLoader worker: PID {child.pid}")
                    child.terminate()
                    terminated_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        if terminated_count > 0:
            logger.info(f"Terminated {terminated_count} DataLoader worker processes")
            
        # Wait a moment then force kill any remaining
        import time
        time.sleep(1)
        
        for child in current_process.children(recursive=True):
            try:
                if child.is_running():
                    child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
                
    except ImportError:
        logger.warning("psutil not available - cannot cleanup DataLoader processes")
    except Exception as e:
        logger.error(f"Error during DataLoader cleanup: {e}")

def signal_handler(signum, frame):
    """Handle shutdown signals (Ctrl+C, etc.)."""
    logger.info(f"Received signal {signum}, cleaning up...")
    cleanup_dataloader_processes()
    
    if _qt_application:
        _qt_application.quit()
    
    sys.exit(0)

def setup_cleanup_handlers():
    """Set up signal handlers and exit cleanup."""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination
    
    # Register cleanup function to run on normal exit
    atexit.register(cleanup_dataloader_processes)
    
    logger.info("Process cleanup handlers registered")

def setup_multiprocessing():
    """Configure multiprocessing for better worker cleanup."""
    try:
        # Set start method to 'spawn' for better process isolation on Linux
        if platform.system() == 'Linux' and hasattr(multiprocessing, 'set_start_method'):
            try:
                multiprocessing.set_start_method('spawn', force=True)
                logger.info("Set multiprocessing start method to 'spawn' for Linux")
            except RuntimeError:
                # Already set
                pass
                
    except Exception as e:
        logger.warning(f"Could not configure multiprocessing: {e}")

def clear_pycache():
    """Recursively find and remove __pycache__ directories."""
    try:
        logger.info("Starting __pycache__ cleanup...")
        count = 0
        # Start from the directory of the launch script
        start_dir = os.path.dirname(os.path.abspath(__file__))
        for root, dirs, files in os.walk(start_dir):
            if '__pycache__' in dirs:
                pycache_path = os.path.join(root, '__pycache__')
                shutil.rmtree(pycache_path)
                count += 1
        if count > 0:
            logger.info(f"Removed {count} __pycache__ directories.")
    except Exception as e:
        logger.warning(f"Could not clear __pycache__: {e}")

def main():
    global _qt_application
    
    # Clean up __pycache__ directories on startup
    clear_pycache()
    
    # Configure multiprocessing first
    setup_multiprocessing()
    multiprocessing.freeze_support()
    
    # Set up cleanup handlers
    setup_cleanup_handlers()
    
    if '--install-gpu' in sys.argv:
        gpu_setup.install_gpu_pytorch()
        sys.exit(0)
        
    app = QApplication(sys.argv)
    _qt_application = app
    
    # Set application properties for better cleanup
    app.setQuitOnLastWindowClosed(True)
    
    # Initialize configuration manager early
    config_manager = get_config_manager()
    projects_dir = config_manager.get_projects_directory()
    logger.info(f"Vestim starting - Projects directory: {projects_dir}")
    
    # Launch the main welcome screen
    welcome_screen = WelcomeGUI()
    welcome_screen.show()
    
    try:
        # Start the application
        exit_code = app.exec_()
        
        # Cleanup on normal exit
        cleanup_dataloader_processes()
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        cleanup_dataloader_processes()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application error: {e}")
        cleanup_dataloader_processes()
        sys.exit(1)

if __name__ == '__main__':
    main()