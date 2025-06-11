# VEstim Client-Server Architecture Guide

VEstim has been refactored to use a client-server architecture, which provides several benefits:

1. **Resilience**: Training continues even if the GUI crashes
2. **Persistence**: Jobs and tasks are tracked properly
3. **Monitoring**: Multiple jobs can be tracked through the dashboard
4. **Separation of Concerns**: Frontend and backend are properly decoupled

## Installation

Install VEstim with:

```bash
pip install -e .
```

This installs the `vestim` command line tool and all required dependencies.

## Using VEstim

### Starting the Server

Start the VEstim server in the background:

```bash
vestim server
```

Or with development mode enabled (auto-reload on code changes):

```bash
vestim server --reload
```

### Starting the GUI

Start the VEstim dashboard GUI:

```bash
vestim gui
```

This connects to the running server.

### All-in-One Mode

To start both the server and GUI at once:

```bash
vestim all
```

### Checking Server Status

Check if the server is running and view active tasks:

```bash
vestim status
```

### Stopping the Server

Stop the running server:

```bash
vestim stop
```

## Architecture Overview

### Component Separation

1. **Backend Server**: FastAPI-based REST API (vestim.backend)
   - Job management
   - Task execution
   - Data processing
   - ML model training

2. **Frontend GUI**: PyQt5-based user interface (vestim.gui)
   - Job dashboard
   - Data import
   - Data augmentation
   - Training configuration
   - Model evaluation

### Data Flow

1. User imports data through the GUI
2. GUI sends data to the server via REST API
3. Server processes data and creates a job
4. User configures and starts training via GUI
5. Server executes training tasks in the background
6. GUI periodically checks task status and displays progress

### Persistence

- Jobs and tasks are tracked in a registry file
- Training status is stored in JSON files within job folders
- All jobs are stored in the `output` directory

## Developing VEstim

### Adding New Features

1. **Backend Changes**:
   - Add endpoints to `vestim/backend/src/main.py`
   - Add services to `vestim/backend/src/services/`

2. **Frontend Changes**:
   - Add or modify GUIs in `vestim/gui/src/`
   - Update API calls to match backend endpoints

### Best Practices

1. Use the server API for all data operations
2. Keep GUI code focused on presentation only
3. Implement proper error handling in both server and client code
4. Log important events in both client and server

## Troubleshooting

### Server Won't Start

1. Check if a server is already running: `vestim status`
2. Look for port conflicts or permission issues
3. Check the server logs in `~/.vestim/server.log`

### GUI Can't Connect to Server

1. Verify the server is running: `vestim status`
2. Check network settings if running on a remote server
3. Ensure firewall is not blocking the connection

### Training Tasks Failing

1. Check server logs
2. Review job-specific logs in the job folder
3. Ensure all dependencies are properly installed

## Support

For issues or questions, please file a bug report or contact the development team.
