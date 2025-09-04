import sys
import multiprocessing
from PyQt5.QtWidgets import QApplication
from vestim.gui.src.welcome_gui_qt import WelcomeGUI
from vestim.utils import gpu_setup
from vestim.config_manager import get_config_manager
import logging

# Set up a logger for the launcher
logger = logging.getLogger(__name__)

def main():
    multiprocessing.freeze_support()
    
    if '--install-gpu' in sys.argv:
        gpu_setup.install_gpu_pytorch()
        sys.exit(0)
        
    app = QApplication(sys.argv)
    
    # Initialize configuration manager early
    config_manager = get_config_manager()
    projects_dir = config_manager.get_projects_directory()
    logger.info(f"Vestim starting - Projects directory: {projects_dir}")
    
    # Launch the main welcome screen
    welcome_screen = WelcomeGUI()
    welcome_screen.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()