# VEstim Quick Start Guide

This guide will help you quickly get started with VEstim, a machine learning model training tool with a client-server architecture.

## Starting VEstim

VEstim can be started in several ways:

### Option 1: Using the Launcher Script (Recommended)

Simply run the `launch_vestim.cmd` script in the root directory:

```
launch_vestim.cmd
```

This will start both the server and GUI components automatically.

### Option 2: Using the Command Line

VEstim provides several commands through its entry point:

```
# Start both server and GUI (default)
python -m vestim.scripts.entrypoint

# Start only the server
python -m vestim.scripts.entrypoint server

# Start only the GUI
python -m vestim.scripts.entrypoint gui

# Check server status
python -m vestim.scripts.entrypoint status

# Stop the server
python -m vestim.scripts.entrypoint stop
```

### Option 3: Starting Components Separately

You can also start the server and GUI separately:

```
# Start the server
python -m vestim.scripts.run_server

# Start the GUI
python -m vestim.scripts.run_gui
```

## Troubleshooting

### Server Won't Start

If you see an error about the port being in use:

1. Check if the server is already running with: `python -m vestim.scripts.entrypoint status`
2. If it's running but you need to restart it, use: `python -m vestim.scripts.entrypoint stop`
3. Then start it again: `python -m vestim.scripts.entrypoint server`

### GUI Can't Connect to Server

The GUI will automatically attempt to start the server if it's not running. If this fails:

1. Make sure there are no firewall restrictions blocking port 8001
2. Check if another application is using port 8001
3. Try starting the server manually before the GUI

### Job Management

From the dashboard, you can:

1. Create new jobs
2. Monitor existing jobs
3. Stop running tasks
4. Clear job history

## Architecture Overview

VEstim uses a client-server architecture:

- **Server**: Handles all computational tasks and persists job information
- **GUI**: Provides a user interface to manage and monitor jobs
- **Data Flow**: All job requests go through the REST API provided by the server

The server continues running even if the GUI is closed, allowing long-running tasks to complete.
