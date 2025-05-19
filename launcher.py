#!/usr/bin/env python3
"""
VESTim Launcher Script
======================
This script handles display configuration and launches the VESTim application.
It provides options for different display modes and helps with remote usage.
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path

# Add the project root to the Python path to ensure imports work
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

try:
    from vestim.config.display_config import DisplayConfig
except ImportError:
    print("Error: Could not import VESTim modules. Check your installation.")
    sys.exit(1)

def setup_logging():
    """Set up logging configuration."""
    log_dir = os.path.join(os.path.expanduser("~"), ".vestim", "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "vestim_launcher.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("vestim_launcher")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Launch VESTim application')
    
    parser.add_argument('--display', help='Set the DISPLAY environment variable (e.g., 192.168.1.5:0.0)')
    parser.add_argument('--config', help='Path to custom config file')
    parser.add_argument('--create-config', action='store_true', help='Create default configuration file')
    parser.add_argument('--no-gui', action='store_true', help='Run in non-GUI mode')
    parser.add_argument('--vnc', action='store_true', help='Start a VNC server for remote viewing')
    parser.add_argument('--vnc-port', type=int, default=5901, help='Port for VNC server')
    parser.add_argument('--test-display', action='store_true', help='Test if display is working correctly')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                      help='Set the logging level')
    
    return parser.parse_args()

def test_display():
    """Test if display is working by launching a simple test window."""
    try:
        # Try to import a GUI library that should be part of the installation
        display_test_code = """
import tkinter as tk
root = tk.Tk()
root.title("Display Test")
label = tk.Label(root, text="Display is working correctly!")
label.pack(padx=20, pady=20)
root.after(3000, root.destroy)  # Close after 3 seconds
root.mainloop()
"""
        result = subprocess.run([sys.executable, '-c', display_test_code], 
                               stderr=subprocess.PIPE, timeout=10)
        
        if result.returncode == 0:
            return True
        else:
            logger.error(f"Display test failed: {result.stderr.decode()}")
            return False
    except Exception as e:
        logger.error(f"Display test error: {str(e)}")
        return False

def start_vnc_server(port):
    """Start a VNC server for remote display."""
    try:
        # Check if a VNC server is available
        vnc_check = subprocess.run(['which', 'x11vnc'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if vnc_check.returncode != 0:
            logger.error("x11vnc not found. Please install it (e.g., apt-get install x11vnc)")
            return False
            
        # Start VNC server with automatic display detection
        vnc_cmd = ['x11vnc', '-create', '-forever', '-bg', '-rfbport', str(port), '-noxdamage', '-nopw']
        
        logger.info(f"Starting VNC server on port {port}")
        subprocess.Popen(vnc_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        logger.info(f"VNC server started. Connect to this machine on port {port}")
        logger.info("To connect from a client, use a VNC viewer pointed at:")
        hostname = subprocess.check_output(['hostname'], text=True).strip()
        logger.info(f"    {hostname}:{port} or ip_address:{port}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to start VNC server: {str(e)}")
        return False

def launch_application(no_gui=False):
    """Launch the main VESTim application."""
    try:
        if no_gui:
            # Import and run the command-line version
            logger.info("Launching in command-line mode")
            from vestim.gateway.src import cli_app
            cli_app.main()
        else:
            # Import and run the GUI version
            logger.info("Launching GUI application")
            from vestim.gui.src import main_gui
            main_gui.main()
            
        return True
    except Exception as e:
        logger.error(f"Failed to launch application: {str(e)}")
        return False

if __name__ == "__main__":
    # Set up logging first
    logger = setup_logging()
    args = parse_arguments()
    
    # Set logging level from args
    logger.setLevel(getattr(logging, args.log_level))
    
    # Create default config if requested
    if args.create_config:
        display_config = DisplayConfig(args.config)
        config_path = display_config.create_default_config()
        logger.info(f"Created default configuration at {config_path}")
        sys.exit(0)
    
    # Handle display configuration
    if args.display:
        # Use command-line provided display
        os.environ['DISPLAY'] = args.display
        logger.info(f"Using command line DISPLAY: {args.display}")
    else:
        # Use automatic display configuration
        display_config = DisplayConfig(args.config)
        display_setup_success = display_config.setup_display()
        
        if not display_setup_success:
            logger.warning("Display setup unsuccessful. GUI functionality may be limited.")
            
            if not args.no_gui:
                logger.warning("Consider using --no-gui mode or configure display manually.")
    
    # Test display if requested
    if args.test_display:
        if test_display():
            logger.info("Display test successful!")
        else:
            logger.error("Display test failed. Check your X11 configuration.")
        sys.exit(0)
    
    # Start VNC server if requested
    if args.vnc:
        vnc_started = start_vnc_server(args.vnc_port)
        if not vnc_started:
            logger.error("Failed to start VNC server. Continuing with normal launch.")
    
    # Launch the application
    launch_success = launch_application(args.no_gui)
    
    if launch_success:
        logger.info("Application launched successfully")
        sys.exit(0)
    else:
        logger.error("Application failed to launch")
        sys.exit(1)
