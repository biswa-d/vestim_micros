#!/usr/bin/env python3
"""
Standalone Testing GUI Launcher for VEstim Tool
This script launches the comprehensive standalone testing GUI for testing individual models.
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import and run the GUI
from vestim.gui.src.standalone_testing_gui_qt import main

if __name__ == "__main__":
    print("VEstim Comprehensive Standalone Testing GUI")
    print("=" * 60)
    print("Starting comprehensive standalone testing interface...")
    print("Features:")
    print("- Training metrics display (epochs, losses)")
    print("- Testing performance metrics (MAE, RMSE, R², etc.)")  
    print("- Real-time visualization plots")
    print("- Results history and persistence")
    print("- Professional user interface")
    print("=" * 60)
    
    try:
        main()
    except Exception as e:
        print(f"Error starting GUI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)