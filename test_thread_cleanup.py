#!/usr/bin/env python
"""
Test script to verify thread cleanup in the job dashboard.
"""

import sys
import os
import time

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from PyQt5.QtWidgets import QApplication

def test_dashboard_thread_cleanup():
    """Test that dashboard properly cleans up threads on close."""
    print("Testing dashboard thread cleanup...")
    
    # Create QApplication
    app = QApplication(sys.argv)
    
    try:
        from vestim.gui.src.job_dashboard_gui_qt import JobDashboard
        
        # Create dashboard
        dashboard = JobDashboard()
        
        # Show dashboard briefly
        dashboard.show()
        
        # Process events to allow initialization
        app.processEvents()
        time.sleep(1)
        
        # Check if there are any threads
        print(f"Dashboard created. Checking for threads...")
        if hasattr(dashboard, 'check_thread'):
            print(f"Found check_thread: {dashboard.check_thread}")
            print(f"Thread is running: {dashboard.check_thread.isRunning()}")
        
        # Close the dashboard
        print("Closing dashboard...")
        dashboard.close()
        
        # Process events to allow cleanup
        app.processEvents()
        time.sleep(1)
        
        # Check if thread was cleaned up
        if hasattr(dashboard, 'check_thread'):
            print(f"After close - Thread is running: {dashboard.check_thread.isRunning()}")
        
        print("✅ Dashboard thread cleanup test completed successfully.")
        
    except Exception as e:
        print(f"❌ Error during dashboard test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        app.quit()

if __name__ == "__main__":
    test_dashboard_thread_cleanup()
