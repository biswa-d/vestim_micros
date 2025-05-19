"""
Configuration module for display settings and X11 forwarding.
This allows users to customize their display settings for remote usage.
"""
import os
import socket
import configparser
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DisplayConfig:
    def __init__(self, config_file=None):
        """Initialize display configuration.
        
        Args:
            config_file (str, optional): Path to the config file.
                If None, will look for .vestimrc in user's home directory.
        """
        self.config_file = config_file or os.path.expanduser("~/.vestimrc")
        self.config = configparser.ConfigParser()
        self.display_settings = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from the config file."""
        if os.path.exists(self.config_file):
            try:
                self.config.read(self.config_file)
                if 'Display' in self.config:
                    self.display_settings = dict(self.config['Display'])
                    logger.info(f"Loaded display settings from {self.config_file}")
                else:
                    logger.warning(f"No Display section found in {self.config_file}")
            except Exception as e:
                logger.error(f"Error loading config: {str(e)}")
        else:
            logger.info(f"Config file {self.config_file} not found. Using defaults.")
    
    def setup_display(self):
        """Set up the display environment based on configuration.
        
        Returns:
            bool: True if display was set successfully, False otherwise
        """
        # Check if DISPLAY is already set properly (e.g., by SSH -X)
        if 'DISPLAY' in os.environ and self._test_display():
            logger.info(f"Using existing DISPLAY: {os.environ['DISPLAY']}")
            return True
            
        # Use configuration file if available
        if 'host' in self.display_settings:
            display_str = f"{self.display_settings.get('host')}:{self.display_settings.get('display_number', '0')}.{self.display_settings.get('screen_number', '0')}"
            os.environ['DISPLAY'] = display_str
            logger.info(f"Set DISPLAY to {display_str} from config file")
            return self._test_display()
            
        # No configuration found, try to determine automatically
        try:
            # Try to get the IP of the SSH client if this is an SSH session
            if 'SSH_CLIENT' in os.environ:
                client_ip = os.environ['SSH_CLIENT'].split()[0]
                display_str = f"{client_ip}:0.0"
                os.environ['DISPLAY'] = display_str
                logger.info(f"Set DISPLAY to {display_str} based on SSH client")
                return self._test_display()
        except Exception as e:
            logger.warning(f"Failed to determine display: {str(e)}")
            
        logger.error("Could not set up display. Please configure it manually.")
        return False
    
    def _test_display(self):
        """Test if the display is working.
        
        Returns:
            bool: True if display is working, False otherwise
        """
        try:
            # Simply check if DISPLAY is set, more comprehensive tests 
            # could be done with a small test window but would require 
            # Qt/Tk dependencies
            return 'DISPLAY' in os.environ and os.environ['DISPLAY'] != ""
        except Exception as e:
            logger.error(f"Display test failed: {str(e)}")
            return False
    
    def create_default_config(self):
        """Create a default configuration file."""
        if not os.path.exists(os.path.dirname(self.config_file)):
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
        self.config['Display'] = {
            'host': '127.0.0.1',
            'display_number': '0',
            'screen_number': '0',
            '# Uncomment and set these if you need custom X11 settings': '',
            '# xauth_protocol': 'MIT-MAGIC-COOKIE-1',
            '# use_xhost': 'False'
        }
        
        with open(self.config_file, 'w') as f:
            self.config.write(f)
        
        logger.info(f"Created default config at {self.config_file}")
        
        # Also create instructions file
        instructions_path = os.path.join(os.path.dirname(self.config_file), "display_setup_instructions.txt")
        with open(instructions_path, 'w') as f:
            f.write("""
VESTIM DISPLAY CONFIGURATION INSTRUCTIONS
=========================================

To run VESTim GUI remotely with proper display forwarding, use one of these methods:

1. SSH X11 Forwarding (Recommended)
   -----------------------------
   Connect to the remote server with:
   $ ssh -X username@remote_server
   
   Or for trusted forwarding (better performance):
   $ ssh -Y username@remote_server
   
   This will automatically set up DISPLAY. No further configuration needed.

2. Manual Configuration
   -------------------
   Edit ~/.vestimrc and set your local machine's IP in the [Display] section:
   
   [Display]
   host = your.local.ip.address
   display_number = 0
   screen_number = 0
   
   Make sure X11 server is running on your local machine:
   - Windows: Use VcXsrv, Xming, or similar
   - Mac: Use XQuartz
   - Linux: Already installed, ensure it allows remote connections with:
     $ xhost +

3. Using the VNC Option
   ------------------
   VESTim also supports running with VNC. Install a VNC server on the remote machine,
   start it, and connect from your local machine.
   
For more help, consult the documentation or contact support.
""")
        logger.info(f"Created instructions at {instructions_path}")
        
        return self.config_file
        
# Example usage in application startup code:
# display_config = DisplayConfig()
# if display_config.setup_display():
#     # Continue with GUI initialization
# else:
#     # Provide error message or fallback to text mode
