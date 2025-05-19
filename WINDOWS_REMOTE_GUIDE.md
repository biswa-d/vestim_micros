# VESTim GPU Installation and Usage Guide

This guide will help you install and run VESTim on a remote Linux server with GPU support, 
while accessing the GUI from your Windows machine.

## Prerequisites

- A Windows machine with:
  - PuTTY or OpenSSH client
  - An X server for Windows (VcXsrv, Xming, or MobaXterm)
- Access to a Linux server with:
  - GPU support with CUDA
  - Python 3.7 or newer

## Installation Steps

### Step 1: Set Up X Server on Windows

1. **Install VcXsrv** (recommended):
   - Download from [SourceForge](https://sourceforge.net/projects/vcxsrv/)
   - Run the installer
   - Launch XLaunch from the Start menu
   - Use these settings:
     - Multiple windows → Display number: 0 → Start no client
     - **Important**: Check "Disable access control"
   - Click "Next" and then "Finish"

   Or use **MobaXterm** which has a built-in X server.

2. **Note your Windows IP address**:
   - Open Command Prompt
   - Type `ipconfig` and look for your IPv4 Address

### Step 2: Install VESTim on the Remote Server

1. **Connect to the remote server** with SSH and X11 forwarding:
   
   Using PuTTY:
   - Set Host Name to your server
   - Navigate to Connection > SSH > X11
   - Check "Enable X11 forwarding"
   - Click Open and log in

   Using Command Line (PowerShell or OpenSSH):
   ```
   ssh -X username@your_server_address
   ```

2. **Install VESTim package**:
   ```bash
   # Create and activate a Python virtual environment (recommended)
   python -m venv vestim-env
   source vestim-env/bin/activate

   # Install the package from the provided file
   pip install vestim-reg-gpu-1.0.0.tar.gz  # Filename may vary
   ```

### Step 3: Configure Display Settings

1. **Create a configuration file** (if not already done by the installation):
   ```bash
   python -m vestim.config.create_config
   ```

2. **Edit the configuration file**:
   ```bash
   nano ~/.vestimrc
   ```

3. **Set your Windows IP address** in the `[Display]` section:
   ```
   [Display]
   host = YOUR_WINDOWS_IP
   display_number = 0
   screen_number = 0
   ```

### Step 4: Run VESTim

You can run VESTim in several ways:

- **Full application**:
  ```bash
  vestim
  ```

- **Specific components**:
  ```bash
  vestim-gui --mode data_import  # Data Import GUI
  vestim-gui --mode training     # Training GUI
  vestim-gui --mode testing      # Testing GUI
  ```

- **Original method** (direct module execution):
  ```bash
  python -m vestim.gui.src.data_import_gui_qt
  ```

- **With explicit display setting**:
  ```bash
  vestim --display=YOUR_WINDOWS_IP:0.0
  ```

## Troubleshooting

### No GUI Window Appears

1. **Check X Server**: Make sure your X server is running on Windows

2. **Test X11 Forwarding**:
   ```bash
   echo $DISPLAY  # Should show something like YOUR_IP:0.0
   xeyes          # Try running a simple X application
   ```

3. **Firewall Issues**: Ensure your Windows firewall isn't blocking X11 (typically port 6000)

4. **Try VNC Mode**: If X11 forwarding doesn't work:
   ```bash
   vestim --vnc --vnc-port=5900
   ```
   Then connect using a VNC client like TightVNC Viewer to your server on port 5900

### Performance Issues

- Add compression to your SSH connection: `ssh -X -C username@server`
- Consider using a VNC connection for better performance with complex visualizations

## Getting Help

If you encounter any issues not covered in this guide, please contact:
- Email: biswanath.dehury@mcmaster.ca
