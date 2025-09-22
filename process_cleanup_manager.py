#!/usr/bin/env python3
"""
Process Cleanup Manager for VEstim
Handles proper cleanup of multiprocessing workers and child processes.
"""

import os
import sys
import signal
import atexit
import psutil
import logging
from typing import List, Set
import threading
import time

class ProcessCleanupManager:
    """Manages cleanup of child processes and worker threads."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.child_processes: Set[int] = set()
        self.cleanup_thread = None
        self.shutdown_event = threading.Event()
        self.registered = False
        
    def register_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        if self.registered:
            return
            
        try:
            # Register signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            if hasattr(signal, 'SIGHUP'):
                signal.signal(signal.SIGHUP, self._signal_handler)
            
            # Register atexit handler
            atexit.register(self.cleanup_all_processes)
            
            self.registered = True
            self.logger.info("Process cleanup manager registered signal handlers")
            
        except Exception as e:
            self.logger.error(f"Failed to register signal handlers: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating cleanup...")
        self.shutdown_event.set()
        self.cleanup_all_processes()
        sys.exit(0)
    
    def track_child_process(self, pid: int):
        """Track a child process for cleanup."""
        self.child_processes.add(pid)
        self.logger.debug(f"Tracking child process: {pid}")
    
    def cleanup_all_processes(self):
        """Clean up all tracked child processes and PyTorch workers."""
        try:
            self.logger.info("Starting comprehensive process cleanup...")
            
            # Get current process and its children
            current_process = psutil.Process()
            main_pid = current_process.pid
            
            # Find all child processes
            all_children = []
            try:
                all_children = current_process.children(recursive=True)
            except psutil.NoSuchProcess:
                pass
            
            # Find processes by name (launch_gui_qt workers)
            launch_gui_processes = []
            try:
                for proc in psutil.process_iter(['pid', 'ppid', 'name', 'cmdline']):
                    try:
                        if proc.info['cmdline'] and any('launch_gui_qt' in cmd for cmd in proc.info['cmdline']):
                            if proc.info['pid'] != main_pid:  # Don't kill main process
                                launch_gui_processes.append(proc)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            except Exception as e:
                self.logger.warning(f"Error finding launch_gui_qt processes: {e}")
            
            # Combine all processes to cleanup
            processes_to_cleanup = set()
            
            # Add tracked child processes
            for pid in self.child_processes:
                try:
                    processes_to_cleanup.add(psutil.Process(pid))
                except psutil.NoSuchProcess:
                    pass
            
            # Add all child processes
            processes_to_cleanup.update(all_children)
            
            # Add launch_gui_qt processes (but not main process)
            processes_to_cleanup.update(launch_gui_processes)
            
            if not processes_to_cleanup:
                self.logger.info("No child processes found to cleanup")
                return
            
            self.logger.info(f"Found {len(processes_to_cleanup)} child processes to cleanup")
            
            # First, try graceful shutdown
            for proc in processes_to_cleanup:
                try:
                    if proc.is_running():
                        self.logger.debug(f"Terminating process {proc.pid} ({proc.name()})")
                        proc.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    self.logger.debug(f"Could not terminate process {proc.pid}: {e}")
            
            # Wait a bit for graceful shutdown
            time.sleep(2)
            
            # Force kill any remaining processes
            for proc in processes_to_cleanup:
                try:
                    if proc.is_running():
                        self.logger.warning(f"Force killing process {proc.pid} ({proc.name()})")
                        proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    self.logger.debug(f"Could not kill process {proc.pid}: {e}")
            
            # Clear tracked processes
            self.child_processes.clear()
            self.logger.info("Process cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during process cleanup: {e}")
    
    def start_monitoring(self):
        """Start background monitoring of child processes."""
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            return
            
        self.cleanup_thread = threading.Thread(target=self._monitor_processes, daemon=True)
        self.cleanup_thread.start()
        self.logger.info("Started process monitoring thread")
    
    def _monitor_processes(self):
        """Monitor and clean up zombie processes."""
        while not self.shutdown_event.is_set():
            try:
                # Check for zombie processes every 30 seconds
                time.sleep(30)
                
                # Clean up zombie processes
                current_process = psutil.Process()
                for child in current_process.children():
                    try:
                        if child.status() == psutil.STATUS_ZOMBIE:
                            self.logger.warning(f"Cleaning up zombie process {child.pid}")
                            child.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                        
            except Exception as e:
                self.logger.error(f"Error in process monitoring: {e}")
    
    def cleanup_pytorch_workers(self):
        """Specifically clean up PyTorch DataLoader worker processes."""
        try:
            current_process = psutil.Process()
            
            for child in current_process.children(recursive=True):
                try:
                    # Check if it's a Python worker process
                    cmdline = child.cmdline()
                    if cmdline and len(cmdline) > 0:
                        if ('python' in cmdline[0].lower() and 
                            any('dataloader' in arg.lower() or 'worker' in arg.lower() for arg in cmdline)):
                            self.logger.info(f"Terminating PyTorch worker process {child.pid}")
                            child.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                    
        except Exception as e:
            self.logger.error(f"Error cleaning up PyTorch workers: {e}")

# Global instance
_cleanup_manager = None

def get_cleanup_manager() -> ProcessCleanupManager:
    """Get the global cleanup manager instance."""
    global _cleanup_manager
    if _cleanup_manager is None:
        _cleanup_manager = ProcessCleanupManager()
        _cleanup_manager.register_signal_handlers()
        _cleanup_manager.start_monitoring()
    return _cleanup_manager

def cleanup_on_exit():
    """Convenience function for cleanup on exit."""
    manager = get_cleanup_manager()
    manager.cleanup_all_processes()