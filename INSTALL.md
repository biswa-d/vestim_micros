# VESTim Installation and Usage Guide

This guide will help you install and use the VESTim package on a remote server with GPU support.

## Prerequisites

- Linux server with CUDA support
- Python 3.7 or higher
- SSH access to the server (ideally with X11 forwarding capability)
- An X11 server on your local machine:
  - Windows: VcXsrv, Xming, or MobaXterm
  - macOS: XQuartz
  - Linux: already installed

## Installation Options

### Option 1: Install from Source (Recommended for Development)

1. Clone the repository on the remote server:
   ```bash
   git clone <repository-url> vestim-gpu
   cd vestim-gpu
   ```

2. Create a virtual environment and install the package:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -e .
   ```

3. Create a default configuration file:
   ```bash
   python launcher.py --create-config
   ```

### Option 2: Install from the Provided Package (Easiest)

1. Copy the provided wheel file to the remote server.

2. Install using pip:
   ```bash
   pip install vestim-reg-gpu-X.Y.Z-py3-none-any.whl
   ```

3. Create a default configuration file:
   ```bash
   vestim --create-config
   ```

## Display Configuration Options

VESTim needs to display its GUI, which can be challenging when running on a remote server. Here are several ways to handle this:

### Option 1: SSH with X11 Forwarding (Recommended)

This is the easiest method if your SSH client supports X11 forwarding.

1. Connect to the remote server with X11 forwarding:
   ```bash
   ssh -X username@remote-server
   # or for trusted forwarding with better performance:
   ssh -Y username@remote-server
   ```

2. Launch VESTim:
   ```bash
   vestim-gui
   ```

The GUI will automatically display on your local machine.

### Option 2: Manual Display Configuration

If X11 forwarding doesn't work or isn't available:

1. Start an X11 server on your local machine.
   - Windows: Start VcXsrv or Xming
   - macOS: Start XQuartz
   - Linux: Enable connections with `xhost +`

2. Find your local machine's IP address.

3. Edit the configuration file on the remote server:
   ```bash
   nano ~/.vestimrc
   ```

4. Set your local machine's IP in the [Display] section:
   ```
   [Display]
   host = your.local.ip.address
   display_number = 0
   screen_number = 0
   ```

5. Launch VESTim with the configured display:
   ```bash
   vestim-gui
   ```

### Option 3: Use VNC (Most Reliable)

VNC works through firewalls and doesn't require X11.

1. On the remote server, launch VESTim with VNC:
   ```bash
   vestim --vnc --vnc-port 5901
   ```

2. On your local machine, connect to the remote server using a VNC client:
   - Use the server's IP address and port 5901
   - For example: `192.168.1.100:5901`

## Testing Your Display Configuration

You can test if your display configuration is working without starting the full application:

```bash
vestim --test-display
```

This will show a simple window if the display is configured correctly.

## Troubleshooting

1. **"Cannot connect to X server" error**:
   - Check that X11 forwarding is enabled in your SSH connection
   - Verify your local X11 server is running
   - Check firewall settings on both machines
   - Try manually setting the DISPLAY variable: `vestim-gui --display your.ip.address:0.0`

2. **GUI appears but looks incomplete or crashes**:
   - Try using VNC mode with `vestim --vnc`
   - If using X11 forwarding, try the trusted mode: `ssh -Y` instead of `ssh -X`

3. **CUDA/GPU errors**:
   - Verify CUDA is properly installed: `nvidia-smi`
   - Check PyTorch can see the GPU: `python -c "import torch; print(torch.cuda.is_available())"`

## For More Help

If you encounter issues, please check the logs located at:
- `~/.vestim/logs/vestim_launcher.log`
- `~/.vestim/logs/vestim_gui.log`

Contact the developer (Biswanath Dehury) for additional assistance.
