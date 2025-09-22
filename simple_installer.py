#!/usr/bin/env python3
"""
PyBattML Simple Installer - launches the main GUI directly
"""

import sys
import os
from pathlib import Path

# Add vestim to path if running from PyInstaller bundle
if hasattr(sys, '_MEIPASS'):
    vestim_path = Path(sys._MEIPASS)
    if str(vestim_path) not in sys.path:
        sys.path.insert(0, str(vestim_path))

# Import and run the main application
if __name__ == "__main__":
    # Import the launch script
    import launch_gui_qt
    # The launch_gui_qt script will handle everything