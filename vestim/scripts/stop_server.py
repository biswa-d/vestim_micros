import psutil
import os
import signal

def stop_existing_server():
    """
    Finds and terminates any existing VEstim server processes.
    """
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Check if the process is a python process running the vestim server
            if 'python' in proc.info['name'].lower() and any('run_server' in part for part in proc.info['cmdline']):
                print(f"Found running server process {proc.info['pid']}. Terminating...")
                os.kill(proc.info['pid'], signal.SIGTERM)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

if __name__ == "__main__":
    stop_existing_server()