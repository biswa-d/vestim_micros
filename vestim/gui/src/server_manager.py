import os
import signal
import platform
import psutil
import logging
import time
from typing import List, Dict, Any, Optional
import requests

class ServerManager:
    """
    Singleton class to manage the VEstim server process.
    Ensures the server process is properly tracked and can be terminated when needed.
    """
    _instance = None
    _server_process = None
    _server_log = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ServerManager, cls).__new__(cls)
            cls._instance.logger = logging.getLogger(__name__)
        return cls._instance
    
    @classmethod
    def check_server_running(cls):
        """Check if the server is running and accessible."""
        try:
            # Try both localhost and 0.0.0.0
            endpoints = ["http://127.0.0.1:8001", "http://localhost:8001"]
            
            for base_url in endpoints:
                try:
                    # Try health endpoint
                    response = requests.get(f"{base_url}/health", timeout=2)
                    if response.status_code == 200:
                        # Update the APIGateway base_url if needed
                        try:
                            from vestim.gui.src.api_gateway import APIGateway
                            api_gateway = APIGateway()
                            if api_gateway.base_url != base_url:
                                cls._instance.logger.info(f"Updating APIGateway base_url to {base_url}")
                                api_gateway.base_url = base_url
                        except Exception as e:
                            cls._instance.logger.error(f"Failed to update APIGateway base_url: {e}")
                        
                        return True
                except:
                    # Try root endpoint
                    try:
                        response = requests.get(f"{base_url}/", timeout=2)
                        if response.status_code == 200:
                            # Update the APIGateway base_url if needed
                            try:
                                from vestim.gui.src.api_gateway import APIGateway
                                api_gateway = APIGateway()
                                if api_gateway.base_url != base_url:
                                    cls._instance.logger.info(f"Updating APIGateway base_url to {base_url}")
                                    api_gateway.base_url = base_url
                            except Exception as e:
                                cls._instance.logger.error(f"Failed to update APIGateway base_url: {e}")
                            
                            return True
                    except:
                        pass
            
            return False
        except Exception as e:
            cls._instance.logger.error(f"Error checking server status: {e}")
            return False
        
    @classmethod
    def set_server_process(cls, process, log_file=None):
        """Set the server process to be managed."""
        # Initialize the instance if needed
        if not hasattr(cls, '_instance') or cls._instance is None:
            cls._instance = cls()
            cls._instance.logger = logging.getLogger(__name__)
            
        # Handle different types of process objects
        if isinstance(process, dict) and 'pid' in process:
            # This is a process dict from psutil
            try:
                # Convert to a psutil Process object if it's just a dict
                cls._server_process = psutil.Process(process['pid'])
                cls._server_log = log_file
                cls._instance.logger.info(f"Server process set with PID: {process['pid']} (from psutil dict)")
            except Exception as e:
                cls._instance.logger.error(f"Failed to set server process from dict: {e}")
                cls._server_process = None
        else:            # Regular subprocess.Popen object or psutil.Process
            cls._server_process = process
            cls._server_log = log_file
            
            if process:
                pid = getattr(process, 'pid', None)
                cls._instance.logger.info(f"Server process set with PID: {pid}")
            else:
                cls._instance.logger.info("Server process set to None")
    
    @classmethod
    def get_server_process(cls):
        """Get the current server process."""
        return cls._server_process
        
    @classmethod
    def terminate_server(cls, api_gateway=None):
        """Terminate the server process if it exists."""
        if not cls._server_process:
            cls._instance.logger.info("No server process to terminate")
            return False
        
        # First, stop all running jobs using the API
        cls.stop_all_jobs(api_gateway)
        
        # Next, force kill any remaining child processes
        cls._kill_child_processes()
            
        try:
            # Handle different process types
            if isinstance(cls._server_process, psutil.Process):
                # If it's a psutil Process object
                pid = cls._server_process.pid
                cls._instance.logger.info(f"Attempting to terminate psutil Process with PID: {pid}")
                
                # Check if process is still running
                if not cls._server_process.is_running():
                    cls._instance.logger.info("Server process has already terminated")
                    cls._cleanup()
                    return True
                
                # Try graceful termination then force kill
                cls._server_process.terminate()
                try:
                    cls._server_process.wait(timeout=5)
                    cls._instance.logger.info("Server process terminated gracefully")
                except:
                    # Force kill if still running
                    if cls._server_process.is_running():
                        cls._instance.logger.info("Force killing server process")
                        cls._server_process.kill()
                
                cls._cleanup()
                return True
            else:
                # Standard subprocess.Popen object
                cls._instance.logger.info(f"Attempting to terminate subprocess.Popen with PID: {cls._server_process.pid}")
                
                # Check if process is still running
                if cls._server_process.poll() is not None:
                    cls._instance.logger.info("Server process has already terminated")
                    cls._cleanup()
                    return True
                
                # Send appropriate signal based on platform
                if platform.system() == "Windows":
                    os.kill(cls._server_process.pid, signal.CTRL_C_EVENT)
                else:
                    os.killpg(os.getpgid(cls._server_process.pid), signal.SIGTERM)
                
                # Wait a bit for graceful shutdown
                try:
                    cls._server_process.wait(timeout=5)
                    cls._instance.logger.info("Server process terminated gracefully")
                except:
                    # Force terminate if still running
                    if cls._server_process.poll() is None:
                        cls._instance.logger.info("Force terminating server process")
                        cls._server_process.terminate()
                        
                        # As a last resort, kill
                        try:
                            cls._server_process.wait(timeout=3)
                        except:
                            if cls._server_process.poll() is None:
                                cls._instance.logger.info("Force killing server process")
                                cls._server_process.kill()
                
                cls._cleanup()
                return True
        except Exception as e:
            cls._instance.logger.error(f"Error terminating server process: {e}")
            
            # Try using psutil as a fallback for non-psutil processes
            try:
                if not isinstance(cls._server_process, psutil.Process):
                    cls._instance.logger.info(f"Attempting to terminate with psutil...")
                    process = psutil.Process(cls._server_process.pid)
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except:
                        if process.is_running():
                            process.kill()
                    return True
            except Exception as e2:
                cls._instance.logger.error(f"Error terminating with psutil: {e2}")
            
            return False
    
    @classmethod
    def _cleanup(cls):
        """Clean up resources."""
        if cls._server_log:
            try:
                cls._server_log.close()
            except:
                pass
        cls._server_process = None
        cls._server_log = None
    
    @classmethod
    def stop_all_jobs(cls, api_gateway=None):
        """
        Stop all running jobs before terminating the server.
        
        Args:
            api_gateway: Optional APIGateway instance to use for stopping jobs
        
        Returns:
            bool: Whether all jobs were successfully stopped
        """
        if not cls._instance:
            cls._instance = cls()
            cls._instance.logger = logging.getLogger(__name__)
        
        cls._instance.logger.info("Stopping all running jobs before server shutdown")
        
        if not api_gateway:
            # Try to import here to avoid circular imports
            try:
                from vestim.gui.src.api_gateway import APIGateway
                api_gateway = APIGateway()
            except Exception as e:
                cls._instance.logger.error(f"Failed to create APIGateway: {e}")
                return False
        
        try:
            # Get all jobs
            jobs = api_gateway.get_all_jobs()
            if not jobs:
                cls._instance.logger.info("No jobs found to stop")
                return True
            
            jobs_stopped = 0
            for job in jobs:
                job_id = job.get("job_id")
                status = job.get("status", "")
                
                if status in ["running", "processing", "training"]:
                    cls._instance.logger.info(f"Stopping job {job_id} with status {status}")
                    try:
                        result = api_gateway.stop_job(job_id)
                        if result.get('status') == 'success':
                            cls._instance.logger.info(f"Successfully stopped job {job_id}")
                            jobs_stopped += 1
                        else:
                            cls._instance.logger.warning(f"Failed to stop job {job_id}: {result.get('message', 'Unknown error')}")
                    except Exception as e:
                        cls._instance.logger.error(f"Error stopping job {job_id}: {e}")
            
            # Log overall results
            cls._instance.logger.info(f"Stopped {jobs_stopped} out of {len(jobs)} jobs")
            
            # Wait a moment for job processes to terminate
            time.sleep(1)
            
            # Return success if we either stopped all running jobs or if there were no running jobs
            return True
        except Exception as e:
            cls._instance.logger.error(f"Error stopping jobs: {e}")
            return False
    
    @classmethod
    def _kill_child_processes(cls):
        """
        Kill all child processes of the server process.
        This ensures that any job processes are terminated before the server is shut down.
        """
        if not cls._server_process:
            cls._instance.logger.info("No server process, so no child processes to terminate")
            return
        
        try:
            parent_pid = None
            
            # Get parent process PID based on the type of process object
            if isinstance(cls._server_process, psutil.Process):
                parent_pid = cls._server_process.pid
            else:
                parent_pid = cls._server_process.pid
            
            # Find all child processes
            parent = psutil.Process(parent_pid)
            children = parent.children(recursive=True)
            
            if not children:
                cls._instance.logger.info("No child processes found")
                return
            
            cls._instance.logger.info(f"Found {len(children)} child processes to terminate")
            
            # First try graceful termination
            for child in children:
                try:
                    cls._instance.logger.info(f"Terminating child process {child.pid}")
                    child.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Wait for them to terminate
            gone, alive = psutil.wait_procs(children, timeout=3)
            cls._instance.logger.info(f"{len(gone)} processes terminated gracefully, {len(alive)} still alive")
            
            # Force kill remaining processes
            for child in alive:
                try:
                    cls._instance.logger.info(f"Force killing child process {child.pid}")
                    child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception as e:
            cls._instance.logger.error(f"Error killing child processes: {e}")
