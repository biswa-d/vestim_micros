#!/usr/bin/env python3
"""
Example configuration setup script for VEstim Tool

This script demonstrates how to set up the default configuration including
the default data directory. This would typically be run during installation
or initial setup.

Usage:
    python setup_config_example.py [data_directory_path]

If no data_directory_path is provided, the user will be prompted to select one.
"""

import os
import sys
from pathlib import Path
from vestim.config_manager import ConfigManager

def setup_default_configuration(data_directory=None):
    """
    Set up the default configuration for VEstim Tool.
    
    Args:
        data_directory (str, optional): Path to the default data directory.
                                      If None, user will be prompted.
    """
    print("Setting up VEstim Tool Configuration...")
    print("-" * 50)
    
    config_manager = ConfigManager()
    
    # Handle data directory
    if data_directory is None:
        print("\nDefault Data Directory Setup:")
        print("The data directory is where VEstim will default to look for")
        print("training, validation, and test data files.")
        print("You can still navigate to other locations when importing data.")
        print()
        
        while True:
            data_dir_input = input("Enter path for default data directory (or press Enter to skip): ").strip()
            
            if not data_dir_input:
                print("Skipping data directory setup. You can set it later.")
                break
            
            # Expand user path and make absolute
            data_dir_path = os.path.abspath(os.path.expanduser(data_dir_input))
            
            # Create directory if it doesn't exist
            if not os.path.exists(data_dir_path):
                try:
                    os.makedirs(data_dir_path, exist_ok=True)
                    print(f"✓ Created data directory: {data_dir_path}")
                except OSError as e:
                    print(f"✗ Could not create directory: {e}")
                    continue
            
            # Set the data directory
            success = config_manager.set_data_directory(data_dir_path)
            if success:
                print(f"✓ Default data directory set to: {data_dir_path}")
                break
            else:
                print("✗ Failed to set data directory. Please try again.")
    else:
        # Data directory provided as argument
        data_dir_path = os.path.abspath(os.path.expanduser(data_directory))
        
        # Create directory if it doesn't exist
        if not os.path.exists(data_dir_path):
            try:
                os.makedirs(data_dir_path, exist_ok=True)
                print(f"✓ Created data directory: {data_dir_path}")
            except OSError as e:
                print(f"✗ Could not create directory: {e}")
                return False
        
        # Set the data directory
        success = config_manager.set_data_directory(data_dir_path)
        if success:
            print(f"✓ Default data directory set to: {data_dir_path}")
        else:
            print("✗ Failed to set data directory.")
            return False
    
    # Display current configuration
    print("\nCurrent Configuration:")
    print("-" * 30)
    output_dir = config_manager.get_output_directory()
    data_dir = config_manager.get_data_directory()
    
    print(f"Output Directory: {output_dir}")
    print(f"Data Directory: {data_dir or 'Not set'}")
    
    print("\nConfiguration setup complete!")
    print("You can modify these settings later by editing:")
    print(f"  {config_manager.config_file}")
    
    return True

def main():
    """Main entry point for the setup script."""
    print("VEstim Tool Configuration Setup")
    print("=" * 50)
    
    # Check if data directory was provided as command line argument
    data_directory = None
    if len(sys.argv) > 1:
        data_directory = sys.argv[1]
        print(f"Using provided data directory: {data_directory}")
    
    # Run the setup
    success = setup_default_configuration(data_directory)
    
    if success:
        print("\n" + "=" * 50)
        print("Setup completed successfully!")
        print("You can now run VEstim Tool with the configured settings.")
    else:
        print("\n" + "=" * 50)
        print("Setup encountered issues. Please check the messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
