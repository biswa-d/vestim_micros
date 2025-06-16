#!/usr/bin/env python
"""
Simple entry point script for VEstim application.
This script launches both the backend server and frontend GUI.
"""

import os
import sys

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

if __name__ == "__main__":
    print("Starting VEstim application...")
    
    # Import and run the main function from the run_vestim module
    try:
        from vestim.scripts.run_vestim import main
        main()
    except ImportError as e:
        print(f"Error importing VEstim modules: {e}")
        print("Make sure you're running this script from the project root directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting VEstim: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
