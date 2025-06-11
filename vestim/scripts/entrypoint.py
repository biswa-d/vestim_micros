#!/usr/bin/env python
# filepath: c:\Users\dehuryb\OneDrive - McMaster University\Models\ML_LiB_Models\vestim_micros\vestim\scripts\entrypoint.py
"""
Unified entry point for the VEstim application.
This script provides access to all functionality through a single command.
"""

import argparse
import os
import sys
import subprocess
import requests
import importlib.util

def verify_installation():
    """
    Verify that all required components are installed.
    """
    try:
        import torch
        import fastapi
        import uvicorn
        import PyQt5
        print("All required dependencies are installed.")
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please run 'pip install -e .' to install all required dependencies.")
        return False

def is_server_running(host="127.0.0.1", port=8001):
    """
    Check if the VEstim server is running.
    """
    try:
        response = requests.get(f"http://{host}:{port}/server/status", timeout=2)
        if response.status_code == 200:
            data = response.json()
            print(f"Server is running with {data['running_tasks']} active tasks.")
            return True
    except requests.RequestException:
        pass
    return False

def start_server(args):
    """
    Start the VEstim server.
    """
    if args.check and is_server_running(args.host, args.port):
        print(f"Server is already running at {args.host}:{args.port}")
        return 0

    # Import run_server module
    run_server = importlib.import_module("vestim.scripts.run_server")
    
    # Run the server
    print(f"Starting VEstim server at {args.host}:{args.port}...")
    return run_server.main()

def start_gui(args):
    """
    Start the VEstim GUI.
    """
    # Import run_gui module
    run_gui = importlib.import_module("vestim.scripts.run_gui")
    
    # Run the GUI
    print("Starting VEstim GUI...")
    return run_gui.main()

def stop_server(args):
    """
    Stop the VEstim server.
    """
    # Import run_server module
    run_server = importlib.import_module("vestim.scripts.run_server")
    
    # Create a parser with the arguments we need
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop", action="store_true", default=True)
    
    # Run the server stop command
    print("Stopping VEstim server...")
    return run_server.main(parser.parse_args([]))

def status(args):
    """
    Check the status of the VEstim server.
    """
    if is_server_running(args.host, args.port):
        print(f"VEstim server is running at {args.host}:{args.port}")
        try:
            # Get more detailed status information
            response = requests.get(f"http://{args.host}:{args.port}/server/status")
            if response.status_code == 200:
                data = response.json()
                print(f"Active tasks: {data['running_tasks']}")
                if data['running_tasks'] > 0:
                    print(f"Task IDs: {', '.join(data['task_ids'])}")
        except requests.RequestException as e:
            print(f"Error getting detailed status: {e}")
        return 0
    else:
        print(f"VEstim server is not running at {args.host}:{args.port}")
        return 1

def main():
    """
    Main entry point for the VEstim command-line interface.
    """
    if not verify_installation():
        return 1

    parser = argparse.ArgumentParser(description="VEstim ML Model Training Tool")
    parser.add_argument("--host", default="127.0.0.1", help="Host address for the server")
    parser.add_argument("--port", type=int, default=8001, help="Port for the server")
    parser.add_argument("--check", action="store_true", help="Check if the server is already running before starting")

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Start the VEstim server")
    server_parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    server_parser.set_defaults(func=start_server)
    
    # GUI command
    gui_parser = subparsers.add_parser("gui", help="Start the VEstim GUI")
    gui_parser.set_defaults(func=start_gui)
    
    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop the VEstim server")
    stop_parser.set_defaults(func=stop_server)
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check the status of the VEstim server")
    status_parser.set_defaults(func=status)
    
    # All command (server + GUI)
    all_parser = subparsers.add_parser("all", help="Start both the server and GUI")
    all_parser.set_defaults(func=lambda args: start_server(args) or start_gui(args))

    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main())
