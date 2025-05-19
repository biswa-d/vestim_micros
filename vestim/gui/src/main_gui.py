#!/usr/bin/env python3
"""
VESTim Main GUI Application
===========================
Main entry point for the VESTim GUI application.
Handles display configuration and launches the appropriate GUI components.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
gui_dir = os.path.dirname(script_dir)
vestim_dir = os.path.dirname(gui_dir)
project_root = os.path.dirname(vestim_dir)
sys.path.insert(0, project_root)

# Import display configuration
try:
    from vestim.config.display_config import DisplayConfig
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from vestim.config.display_config import DisplayConfig
    except ImportError:
        print("Error: Could not import DisplayConfig. Check your installation.")
        sys.exit(1)

# Import PyQt
try:
    from PyQt5.QtWidgets import QApplication
except ImportError:
    print("Error: PyQt5 is required. Install with: pip install PyQt5")
    sys.exit(1)

def setup_logging():
    """Set up logging for the application."""
    log_dir = os.path.join(os.path.expanduser("~"), ".vestim", "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "vestim_gui.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("vestim_gui")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='VESTim GUI Application')
    
    parser.add_argument('--display', help='Set the DISPLAY environment variable (e.g., 192.168.1.5:0.0)')
    parser.add_argument('--config', help='Path to custom config file')
    parser.add_argument('--mode', choices=['training', 'testing', 'data_import', 'all'], 
                      default='all', help='Which GUI component to launch')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                      default='INFO', help='Set the logging level')
    
    return parser.parse_args()

def main():
    """Main entry point for the GUI application."""
    # Set up logging
    logger = setup_logging()
    args = parse_arguments()
    
    # Set logging level
    logger.setLevel(getattr(logging, args.log_level))
    
    # Configure display
    if args.display:
        os.environ['DISPLAY'] = args.display
        logger.info(f"Set DISPLAY to {args.display} from command line")
    else:
        display_config = DisplayConfig(args.config)
        display_setup_success = display_config.setup_display()
        
        if not display_setup_success:
            logger.warning("Display setup unsuccessful. GUI may not work properly.")
            logger.warning("Consider setting DISPLAY manually or using SSH with X11 forwarding.")
    
    # Create QApplication instance
    app = QApplication(sys.argv)
    app.setApplicationName("VESTim")
    
    # Launch appropriate GUI component
    if args.mode == 'training' or args.mode == 'all':
        logger.info("Initializing training setup GUI...")
        from vestim.gui.src.training_setup_gui_qt import VEstimTrainingSetupGUI
        training_gui = VEstimTrainingSetupGUI()
        training_gui.show()
    
    if args.mode == 'testing' or args.mode == 'all':
        logger.info("Initializing testing GUI...")
        from vestim.gui.src.testing_gui_qt import VEstimTestingGUI
        testing_gui = VEstimTestingGUI()
        testing_gui.show()
    
    if args.mode == 'data_import' or args.mode == 'all':
        logger.info("Initializing data import GUI...")
        from vestim.gui.src.data_import_gui_qt import DataImportGUI
        data_import_gui = DataImportGUI()
        data_import_gui.show()
    
    # Start the event loop
    logger.info("Starting PyQt event loop")
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())
