#!/usr/bin/env python3
"""
VESTim Display Configuration Creator
====================================
Helper script to create display configuration files and instructions.
"""

import os
import sys
import logging
import argparse

# Add the project root to the Python path to ensure imports work
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))

try:
    from vestim.config.display_config import DisplayConfig
except ImportError:
    print("Error: Could not import VESTim modules. Check your installation.")
    sys.exit(1)

def setup_logging():
    """Set up basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("vestim_config")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create VESTim display configuration')
    
    parser.add_argument('--config', help='Path to custom config file')
    parser.add_argument('--host', help='Set display host IP (e.g., 192.168.1.5)')
    parser.add_argument('--display', type=int, default=0, help='Set display number')
    parser.add_argument('--screen', type=int, default=0, help='Set screen number')
    
    return parser.parse_args()

def main():
    """Main entry point for creating display configuration."""
    logger = setup_logging()
    args = parse_arguments()
    
    # Create a display config
    display_config = DisplayConfig(args.config)
    
    # Create the default config
    config_path = display_config.create_default_config()
    logger.info(f"Created default configuration at {config_path}")
    
    # Update with command line values if provided
    if args.host:
        # Load the created config
        display_config._load_config()
        
        # Update with command line values
        display_config.config['Display']['host'] = args.host
        display_config.config['Display']['display_number'] = str(args.display)
        display_config.config['Display']['screen_number'] = str(args.screen)
        
        # Save the updated config
        with open(config_path, 'w') as f:
            display_config.config.write(f)
        
        logger.info(f"Updated configuration with host {args.host}:{args.display}.{args.screen}")
    
    logger.info("Configuration file created. You can now run the VESTim application.")
    logger.info("For help with display setup, see the WINDOWS_REMOTE_GUIDE.md file.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
