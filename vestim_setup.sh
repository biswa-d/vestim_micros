#!/bin/bash
# Install and setup script for VEstim
# This script installs VEstim in development mode and provides options to run components

echo "VEstim Setup Script"
echo "================="

if [ -z "$1" ]; then
    echo "Usage: ./vestim_setup.sh [command]"
    echo "Commands:"
    echo "  install - Install VEstim in development mode"
    echo "  server  - Start the VEstim server"
    echo "  gui     - Start the VEstim GUI"
    echo "  all     - Start both server and GUI"
    echo "  stop    - Stop the running server"
    echo "  status  - Check server status"
    echo ""
    echo "Example: ./vestim_setup.sh install"
    exit 0
fi

case "$1" in
    install)
        echo "Installing VEstim in development mode..."
        pip install -e .
        echo "Installation complete."
        ;;
    server)
        echo "Starting VEstim server..."
        vestim server
        ;;
    gui)
        echo "Starting VEstim GUI..."
        vestim gui
        ;;
    all)
        echo "Starting VEstim (server and GUI)..."
        vestim all
        ;;
    stop)
        echo "Stopping VEstim server..."
        vestim stop
        ;;
    status)
        echo "Checking VEstim status..."
        vestim status
        ;;
    *)
        echo "Unknown command: $1"
        echo "Run ./vestim_setup.sh without arguments for help."
        exit 1
        ;;
esac
