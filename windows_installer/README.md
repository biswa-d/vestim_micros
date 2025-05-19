# VESTim Windows Installation Guide

This guide explains how to install and use VESTim on a Windows computer,
connecting to a remote Linux server with GPU support.

## Simple Installation (No Linux Knowledge Required)

### Step 1: Run the Installer

1. Extract the ZIP file to a folder on your computer
2. Double-click `Install-VESTim.bat`
3. Follow the on-screen prompts:
   - The installer will automatically:
     - Install required software (X server for displaying the GUI)
     - Get your server connection details
     - Install VESTim on the remote server
     - Create desktop shortcuts

### Step 2: Launch VESTim

After installation is complete:

1. Double-click the `VESTim` shortcut on your desktop
2. The GUI will appear automatically

That's it! You're now running VESTim with the remote server's GPU power.

## Component-Specific Launchers

The installer also creates separate launchers for individual components:

- `Run-VESTim-data.bat` - Launches only the Data Import component
- `Run-VESTim-train.bat` - Launches only the Training component
- `Run-VESTim-test.bat` - Launches only the Testing component

## Troubleshooting

If you encounter any issues:

1. **No GUI appears**:
   - Make sure your firewall isn't blocking the connection
   - Try running the installer again

2. **Slow performance**:
   - This is normal for X11 forwarding over the internet
   - Consider asking your IT department about VNC or RDP alternatives

3. **Login or connection issues**:
   - Verify your username and password
   - Check that you can connect to the server using regular SSH

For additional help, please contact: biswanath.dehury@mcmaster.ca
