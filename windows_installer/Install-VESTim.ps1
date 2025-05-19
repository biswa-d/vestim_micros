# VESTim Remote Installation PowerShell Script
# This script automates the installation of VESTim on a remote server
# with X11 forwarding for Windows users

# Function to check and install required software
function Install-Prerequisites {
    Write-Host "Checking required software..." -ForegroundColor Cyan
    
    # Check for VcXsrv
    $vcxsrv = Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*" | 
        Where-Object { $_.DisplayName -like "*VcXsrv*" } -ErrorAction SilentlyContinue
    
    if (-not $vcxsrv) {
        Write-Host "VcXsrv X Server not found. Installing..." -ForegroundColor Yellow
        
        # Download VcXsrv installer
        $vcxsrvUrl = "https://sourceforge.net/projects/vcxsrv/files/latest/download"
        $vcxsrvInstaller = "$env:TEMP\vcxsrv_installer.exe"
        
        try {
            Invoke-WebRequest -Uri $vcxsrvUrl -OutFile $vcxsrvInstaller -UseBasicParsing
            
            # Install VcXsrv
            Write-Host "Installing VcXsrv (X Server)..."
            Start-Process -FilePath $vcxsrvInstaller -ArgumentList "/S" -Wait
            
            # Clean up installer
            Remove-Item $vcxsrvInstaller -Force
            
            Write-Host "VcXsrv installed successfully." -ForegroundColor Green
        }
        catch {
            Write-Host "Failed to install VcXsrv automatically." -ForegroundColor Red
            Write-Host "Please install it manually from: https://sourceforge.net/projects/vcxsrv/" -ForegroundColor Red
            Write-Host "Then run this script again." -ForegroundColor Red
            return $false
        }
    }
    else {
        Write-Host "VcXsrv is already installed." -ForegroundColor Green
    }
    
    # Check for OpenSSH
    $ssh = Get-Command ssh -ErrorAction SilentlyContinue
    
    if (-not $ssh) {
        Write-Host "OpenSSH not found. Installing..." -ForegroundColor Yellow
        
        # Install OpenSSH client using Windows capability
        try {
            Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0
            Write-Host "OpenSSH installed successfully." -ForegroundColor Green
        }
        catch {
            Write-Host "Failed to install OpenSSH client automatically." -ForegroundColor Red
            Write-Host "Please enable OpenSSH client in Windows Features, then run this script again." -ForegroundColor Red
            return $false
        }
    }
    else {
        Write-Host "OpenSSH is already installed." -ForegroundColor Green
    }
    
    return $true
}

# Function to get server details from user
function Get-ServerDetails {
    Write-Host "`nServer Connection Details" -ForegroundColor Cyan
    Write-Host "------------------------"
    
    $serverHost = Read-Host "Enter server hostname or IP address"
    $username = Read-Host "Enter your username on the server"
    $password = Read-Host "Enter your password" -AsSecureString
    $port = Read-Host "Enter SSH port (press Enter for default 22)"
    
    if ([string]::IsNullOrEmpty($port)) {
        $port = "22"
    }
    
    # Convert password to plain text for use with plink
    $BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($password)
    $plainPassword = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)
    
    $serverDetails = @{
        "Host" = $serverHost
        "Username" = $username
        "Password" = $plainPassword
        "Port" = $port
    }
    
    return $serverDetails
}

# Function to start VcXsrv with proper settings
function Start-XServer {
    Write-Host "`nStarting X Server..." -ForegroundColor Cyan
    
    # Kill any existing instances
    Stop-Process -Name vcxsrv -ErrorAction SilentlyContinue
    
    # Start VcXsrv with proper parameters
    $xlaunchPath = "$env:ProgramFiles\VcXsrv\xlaunch.exe"
    $xlaunchConfig = "$PSScriptRoot\config.xlaunch"
    
    # Create XLaunch configuration file if it doesn't exist
    if (-not (Test-Path $xlaunchConfig)) {
        @"
<?xml version="1.0" encoding="UTF-8"?>
<XLaunch WindowMode="MultiWindow" ClientMode="NoClient" LocalClient="False" Display="0" LocalProgram="xcalc" RemoteProgram="xterm" RemotePassword="" PrivateKey="" RemoteHost="" RemoteUser="" XDMCPHost="" XDMCPBroadcast="False" XDMCPIndirect="False" Clipboard="True" ClipboardPrimary="True" ExtraParams="" Wgl="True" DisableAC="True" XDMCPTerminate="False"/>
"@ | Out-File -FilePath $xlaunchConfig -Encoding UTF8
    }
    
    # Start VcXsrv with the config file
    Start-Process -FilePath $xlaunchPath -ArgumentList "-run `"$xlaunchConfig`"" -WindowStyle Hidden
    
    # Allow some time for the X server to start
    Start-Sleep -Seconds 2
    
    # Check if VcXsrv is running
    $vcxsrvProcess = Get-Process -Name vcxsrv -ErrorAction SilentlyContinue
    
    if ($vcxsrvProcess) {
        Write-Host "X Server started successfully." -ForegroundColor Green
        return $true
    }
    else {
        Write-Host "Failed to start X Server." -ForegroundColor Red
        return $false
    }
}

# Function to create connection scripts
function Create-ConnectionScripts {
    param (
        [Parameter(Mandatory = $true)]
        [hashtable]$ServerDetails,
        [Parameter(Mandatory = $true)]
        [string]$PackagePath
    )
    
    Write-Host "`nCreating connection scripts..." -ForegroundColor Cyan
    
    # Get local IP address
    $localIP = (Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias $(Get-NetConnectionProfile | Select-Object -ExpandProperty InterfaceAlias) | Where-Object {$_.IPAddress -notmatch '^169'} | Select-Object -ExpandProperty IPAddress)
    
    if (-not $localIP) {
        $localIP = "127.0.0.1"
        Write-Host "Warning: Could not determine local IP address. Using localhost." -ForegroundColor Yellow
    }
    
    # Create the run script
    $runScriptContent = @"
@echo off
:: VESTim Remote Connection Script
:: This script connects to the remote server and runs VESTim

echo Starting X Server...
start "" "%ProgramFiles%\VcXsrv\vcxsrv.exe" -multiwindow -clipboard -wgl -ac

:: Wait for X server to start
timeout /t 2 > nul

:: SSH connection with X11 forwarding
echo Connecting to remote server...
set DISPLAY=$localIP:0.0
ssh -X -o "StrictHostKeyChecking=no" -p $($ServerDetails.Port) $($ServerDetails.Username)@$($ServerDetails.Host) "export DISPLAY=$localIP:0.0 && vestim-gui"

:: Clean up
echo Closing X Server...
taskkill /f /im vcxsrv.exe > nul 2>&1
"@
    
    $runScriptPath = "$PSScriptRoot\Run-VESTim.bat"
    $runScriptContent | Out-File -FilePath $runScriptPath -Encoding ASCII
    
    # Create individual component scripts
    $components = @("data", "train", "test")
    
    foreach ($component in $components) {
        $componentScriptContent = @"
@echo off
:: VESTim $component Component Script
:: This script connects to the remote server and runs the $component component

echo Starting X Server...
start "" "%ProgramFiles%\VcXsrv\vcxsrv.exe" -multiwindow -clipboard -wgl -ac

:: Wait for X server to start
timeout /t 2 > nul

:: SSH connection with X11 forwarding
echo Connecting to remote server...
set DISPLAY=$localIP:0.0
ssh -X -o "StrictHostKeyChecking=no" -p $($ServerDetails.Port) $($ServerDetails.Username)@$($ServerDetails.Host) "export DISPLAY=$localIP:0.0 && vestim-$component"

:: Clean up
echo Closing X Server...
taskkill /f /im vcxsrv.exe > nul 2>&1
"@
        
        $componentScriptPath = "$PSScriptRoot\Run-VESTim-$component.bat"
        $componentScriptContent | Out-File -FilePath $componentScriptPath -Encoding ASCII
    }
    
    # Create the SSH script for remote installation
    $installScriptContent = @"
#!/bin/bash
# VESTim Remote Installation Script

echo "Installing VESTim package..."

# Create Python virtual environment (optional)
python3 -m venv ~/vestim-env
source ~/vestim-env/bin/activate

# Install the package
pip install $PackagePath

# Configure display settings
vestim-config --host $localIP

echo "Installation complete!"
"@
    
    $installScriptPath = "$env:TEMP\vestim_install.sh"
    $installScriptContent | Out-File -FilePath $installScriptPath -Encoding ASCII
    
    Write-Host "Connection scripts created successfully." -ForegroundColor Green
    
    return @{
        "RunScript" = $runScriptPath
        "InstallScript" = $installScriptPath
    }
}

# Function to install VESTim on the remote server
function Install-RemotePackage {
    param (
        [Parameter(Mandatory = $true)]
        [hashtable]$ServerDetails,
        [Parameter(Mandatory = $true)]
        [string]$InstallScript,
        [Parameter(Mandatory = $true)]
        [string]$PackagePath
    )
    
    Write-Host "`nInstalling VESTim on the remote server..." -ForegroundColor Cyan
    
    try {
        # Copy the installation script
        $scpCommand = "scp -o `"StrictHostKeyChecking=no`" -P $($ServerDetails.Port) `"$InstallScript`" $($ServerDetails.Username)@$($ServerDetails.Host):~/"
        Invoke-Expression $scpCommand
        
        # Copy the package file
        $packageName = Split-Path $PackagePath -Leaf
        $scpCommand = "scp -o `"StrictHostKeyChecking=no`" -P $($ServerDetails.Port) `"$PackagePath`" $($ServerDetails.Username)@$($ServerDetails.Host):~/$packageName"
        Invoke-Expression $scpCommand
        
        # Run the installation script
        $sshCommand = "ssh -o `"StrictHostKeyChecking=no`" -p $($ServerDetails.Port) $($ServerDetails.Username)@$($ServerDetails.Host) `"chmod +x ~/vestim_install.sh && ~/vestim_install.sh`""
        Invoke-Expression $sshCommand
        
        Write-Host "VESTim installed successfully on the remote server." -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "Failed to install VESTim on the remote server: $_" -ForegroundColor Red
        return $false
    }
}

# Main installation function
function Install-VESTim {
    # Clear the console
    Clear-Host
    
    Write-Host "===========================================" -ForegroundColor Green
    Write-Host "   VESTim Remote Installation Assistant   " -ForegroundColor Green
    Write-Host "===========================================" -ForegroundColor Green
    Write-Host
    Write-Host "This assistant will help you set up VESTim on a remote server" -ForegroundColor Cyan
    Write-Host "with GPU support, while allowing you to use the GUI from your" -ForegroundColor Cyan
    Write-Host "Windows computer." -ForegroundColor Cyan
    Write-Host
    
    # Install prerequisites
    $prerequisitesOk = Install-Prerequisites
    if (-not $prerequisitesOk) {
        Write-Host "Please install the required software and run this script again." -ForegroundColor Red
        return $false
    }
    
    # Get server details
    $serverDetails = Get-ServerDetails
    
    # Start X Server
    $xServerStarted = Start-XServer
    if (-not $xServerStarted) {
        Write-Host "Please start VcXsrv manually and run this script again." -ForegroundColor Red
        return $false
    }
    
    # Find the package file
    $packageFile = Get-ChildItem -Path "$PSScriptRoot\*.tar.gz" | Select-Object -First 1
    
    if (-not $packageFile) {
        Write-Host "No package file found in the installer directory." -ForegroundColor Red
        Write-Host "Make sure the VESTim package (.tar.gz file) is in the same directory as this script." -ForegroundColor Red
        return $false
    }
    
    # Create connection scripts
    $scripts = Create-ConnectionScripts -ServerDetails $serverDetails -PackagePath "~/$($packageFile.Name)"
    
    # Install VESTim on the remote server
    $installSuccess = Install-RemotePackage -ServerDetails $serverDetails -InstallScript $scripts.InstallScript -PackagePath $packageFile.FullName
    
    if (-not $installSuccess) {
        Write-Host "Failed to install VESTim. Please check the error messages and try again." -ForegroundColor Red
        return $false
    }
    
    # Save server details for future use
    $serverConfig = @"
Host=$($serverDetails.Host)
Username=$($serverDetails.Username)
Port=$($serverDetails.Port)
"@
    
    $serverConfig | Out-File -FilePath "$PSScriptRoot\server_config.txt" -Encoding ASCII
    
    Write-Host "`nInstallation completed successfully!" -ForegroundColor Green
    Write-Host "You can now run VESTim by double-clicking Run-VESTim.bat" -ForegroundColor Green
    
    return $true
}

# Run the installation
Install-VESTim
