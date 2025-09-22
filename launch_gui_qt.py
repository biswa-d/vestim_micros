import sys
import multiprocessing
import os
import signal
import atexit
import platform
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

def main():
    global _qt_application
    
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