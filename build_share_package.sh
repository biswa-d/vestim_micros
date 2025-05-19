#!/bin/bash
# Build VESTim distribution package for sharing

echo "Building VESTim distribution package..."

# Make sure we're in the project root
cd "$(dirname "$0")"

# Ensure Python and required build tools are available
command -v python3 >/dev/null 2>&1 || { echo "Python 3 is required but not found. Aborting."; exit 1; }
command -v pip >/dev/null 2>&1 || { echo "pip is required but not found. Aborting."; exit 1; }

# Install build requirements if needed
pip install --upgrade build setuptools wheel

# Clean up any previous builds
rm -rf dist build *.egg-info

# Build the distribution packages
python -m build

# Create a zip file with the distribution and documentation
mkdir -p share_package
cp dist/*.tar.gz share_package/
cp WINDOWS_REMOTE_GUIDE.md share_package/
cp README.md share_package/
cp requirements.txt share_package/

# Add the Windows installer files to the share package
mkdir -p share_package/windows_installer
cp windows_installer/Install-VESTim.bat share_package/windows_installer/
cp windows_installer/Install-VESTim.ps1 share_package/windows_installer/
cp windows_installer/Run-VESTim.bat share_package/windows_installer/
cp windows_installer/README.md share_package/windows_installer/
cp dist/*.tar.gz share_package/windows_installer/

# Create a simple installation script for convenience
cat > share_package/install_vestim.sh << 'EOL'
#!/bin/bash
# Simple installation script for VESTim

# Extract the package name from the arguments
PACKAGE_FILE=$(ls *.tar.gz 2>/dev/null | head -n 1)

if [ -z "$PACKAGE_FILE" ]; then
    echo "No package file found! Make sure the .tar.gz file is in the same directory."
    exit 1
fi

echo "Found package: $PACKAGE_FILE"

# Ask if the user wants to create a virtual environment
read -p "Create a Python virtual environment? (y/n): " CREATE_VENV

if [[ $CREATE_VENV =~ ^[Yy]$ ]]; then
    # Check if virtualenv is installed
    if ! command -v virtualenv &> /dev/null; then
        echo "virtualenv not found. Installing..."
        pip install virtualenv
    fi
    
    # Create and activate the virtual environment
    virtualenv vestim-env
    source vestim-env/bin/activate
    
    echo "Virtual environment 'vestim-env' created and activated."
else
    echo "Skipping virtual environment creation."
fi

# Install the package
echo "Installing VESTim..."
pip install "$PACKAGE_FILE"

# Create default config
echo "Setting up configuration..."
vestim-config

echo "Installation complete!"
echo "For usage instructions, see WINDOWS_REMOTE_GUIDE.md"
EOL

# Make the installation script executable
chmod +x share_package/install_vestim.sh

# Create zip archive of the share package
zip -r vestim_share_package.zip share_package

echo "Package created: vestim_share_package.zip"
echo "You can now share this package with your colleague."
