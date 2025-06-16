import sys
import os
import time
import subprocess
import signal
import platform
import requests
from PyQt5.QtWidgets import QApplication, QMessageBox, QSplashScreen
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QPainter, QFont
import psutil

# Add the project root to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from vestim.gui.src.job_dashboard_gui_qt import JobDashboard

def start_server():
    """Starts the FastAPI server in a separate process."""
    print("Starting VEstim backend server...")
    
    # Check if server is already running
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name'].lower() and any('vestim.backend.src.main' in cmd for cmd in proc.info['cmdline'] if cmd):
                print(f"Found existing server process with PID: {proc.info['pid']}")
                return None, None
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    
    # Server not running, start it
    command = [sys.executable, "-m", "vestim.backend.src.main"]
    
    # Redirect stdout/stderr to log file
    server_log = open("backend.log", "w")
    
    # Start server process
    try:
        server_process = subprocess.Popen(
            command,
            stdout=server_log,
            stderr=server_log,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if platform.system() == "Windows" else 0,
            start_new_session=True if platform.system() != "Windows" else False
        )
        print(f"Started server with PID: {server_process.pid}")
        return server_process, server_log
    except Exception as e:
        print(f"Error starting server: {e}")
        server_log.close()
        return None, None

def wait_for_server(max_attempts=10, interval=1.0):
    """Wait for server to start responding."""
    print("Waiting for server to start...")
    
    for attempt in range(max_attempts):
        try:
            # Try the health endpoint
            response = requests.get("http://127.0.0.1:8001/health", timeout=2)
            if response.status_code == 200:
                print("Server is up and running!")
                return True
        except:
            # Try the root endpoint as fallback
            try:
                response = requests.get("http://127.0.0.1:8001/", timeout=2)
                if response.status_code == 200:
                    print("Server is up and running!")
                    return True
            except:
                pass
        
        print(f"Waiting for server... ({attempt+1}/{max_attempts})")
        time.sleep(interval)
    
    print("Server did not start correctly after multiple attempts")
    return False

def show_error_and_exit(message):
    """Show error message and exit."""
    app = QApplication.instance() or QApplication(sys.argv)
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setText("VEstim Error")
    msg.setInformativeText(message)
    msg.setWindowTitle("Error")
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec_()
    sys.exit(1)

def main():
    """Main function to start the server and launch the GUI."""
    try:
        # Initialize QApplication
        app = QApplication(sys.argv)
        
        # Create a simple splash screen
        splash_pix = QPixmap(400, 200)
        splash_pix.fill(Qt.white)
        splash = QSplashScreen(splash_pix)
        
        # Draw text on splash screen
        painter = QPainter(splash_pix)
        painter.setFont(QFont("Arial", 12))
        painter.drawText(20, 100, "Starting VEstim...")
        painter.end()
        
        splash.show()
        app.processEvents()
        
        # Start server
        server_process, server_log = start_server()
        
        # Wait for server to start
        if not wait_for_server():
            splash.close()
            show_error_and_exit("Server failed to start. Check backend.log for details.")
            return
            
        # Start dashboard
        dashboard = JobDashboard()
        dashboard.show()
        
        # Close splash screen
        splash.finish(dashboard)
        
        # Run the application
        exit_code = app.exec_()
        
        # Clean up server on exit
        if server_process and server_process.poll() is None:
            print("Stopping server...")
            try:
                if platform.system() == "Windows":
                    os.kill(server_process.pid, signal.CTRL_C_EVENT)
                else:
                    os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
                
                # Wait a bit for graceful shutdown
                server_process.wait(timeout=5)
                
                # Force terminate if needed
                if server_process.poll() is None:
                    server_process.terminate()
            except:
                pass
        
        if server_log:
            server_log.close()
            
        sys.exit(exit_code)
        
    except Exception as e:
        import traceback
        print(f"Error in main: {e}")
        traceback.print_exc()
        show_error_and_exit(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()