#!/usr/bin/env python3
"""
PyInstaller build script for Vestim
Creates a standalone executable bundle
"""

import PyInstaller.__main__
import os
import sys
from pathlib import Path

def build_executable():
    """Build standalone executable using PyInstaller"""
    
    # Get the main script path
    main_script = "vestim/gui/src/data_import_gui_qt.py"
    
    # Get icon path if it exists
    icon_path = "vestim/gui/resources/icon.ico"
    if not Path(icon_path).exists():
        icon_path = None
    
    # PyInstaller arguments
    args = [
        main_script,
        '--name=Vestim',
        '--onefile',  # Create single executable
        # Note: Removed --windowed to show console window with logs
        '--add-data=vestim;vestim',  # Include entire vestim package
        '--add-data=hyperparams.json;.',  # Include config files
        '--hidden-import=PyQt5.QtCore',
        '--hidden-import=PyQt5.QtWidgets', 
        '--hidden-import=PyQt5.QtGui',
        '--hidden-import=pandas',
        '--hidden-import=numpy',
        '--hidden-import=matplotlib',
        '--hidden-import=sklearn',
        '--hidden-import=torch',
        '--hidden-import=scipy',
        '--collect-all=vestim',  # Ensure all vestim modules are included
        '--distpath=dist',
        '--workpath=build',
        '--specpath=.',
    ]
    
    # Add icon if available
    if icon_path:
        args.append(f'--icon={icon_path}')
    
    # Add version info for Windows
    if sys.platform == "win32":
        version_info = create_version_file()
        if version_info:
            args.append(f'--version-file={version_info}')
    
    print("Building executable with PyInstaller...")
    print(f"Arguments: {' '.join(args)}")
    
    # Run PyInstaller
    PyInstaller.__main__.run(args)
    
    print("✓ Executable built successfully!")
    print(f"✓ Output: dist/Vestim.exe")

def create_version_file():
    """Create version file for Windows executable"""
    version_content = '''# UTF-8
#
# For more details about fixed file info 'ffi' see:
# http://msdn.microsoft.com/en-us/library/ms646997.aspx
VSVersionInfo(
  ffi=FixedFileInfo(
# filevers and prodvers should be always a tuple with four items: (1, 2, 3, 4)
# Set not needed items to zero 0.
filevers=(1,0,0,0),
prodvers=(2,0,0,0),
# Contains a bitmask that specifies the valid bits 'flags'r
mask=0x3f,
# Contains a bitmask that specifies the Boolean attributes of the file.
flags=0x0,
# The operating system for which this file was designed.
# 0x4 - NT and there is no need to change it.
OS=0x4,
# The general type of file.
# 0x1 - the file is an application.
fileType=0x1,
# The function of the file.
# 0x0 - the function is not defined for this fileType
subtype=0x0,
# Creation date and time stamp.
date=(0, 0)
),
  kids=[
StringFileInfo(
  [
  StringTable(
    u'040904B0',
    [StringStruct(u'CompanyName', u'Biswanath Dehury'),
    StringStruct(u'FileDescription', u'Vestim - Voltage Estimation Tool for Lithium-ion Batteries'),
    StringStruct(u'FileVersion', u'1.0.0'),
    StringStruct(u'InternalName', u'Vestim'),
    StringStruct(u'LegalCopyright', u'Copyright (c) 2025 Biswanath Dehury'),
    StringStruct(u'OriginalFilename', u'Vestim.exe'),
    StringStruct(u'ProductName', u'Vestim'),
    StringStruct(u'ProductVersion', u'1.0.0')])
  ]), 
VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)'''
    
    version_file = Path("version_info.txt")
    version_file.write_text(version_content)
    return str(version_file)

if __name__ == "__main__":
    try:
        build_executable()
    except Exception as e:
        print(f"Build failed: {e}")
        sys.exit(1)
