"""
Configuration Manager for Vestim
Handles reading installer configuration and setting up directories
"""
import os
import json
import sys
from pathlib import Path

class ConfigManager:
    """Manages configuration for Vestim application"""
    
    def __init__(self):
        self._projects_dir = None
        self._load_config()
    
    def _load_config(self):
        """Load configuration from installer config file"""
        try:
            # Look for config file in application directory
            if getattr(sys, 'frozen', False):
                # Running as compiled executable
                app_dir = Path(sys.executable).parent
            else:
                # Running as script
                app_dir = Path(__file__).parent
            
            config_path = app_dir / "vestim_config.json"
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self._projects_dir = config.get('projects_directory')
                    
                    if self._projects_dir and os.path.exists(self._projects_dir):
                        print(f"Using projects directory from installer: {self._projects_dir}")
                        return
        except Exception as e:
            print(f"Could not load installer config: {e}")
        
        # Fallback to default location
        self._projects_dir = self._get_default_projects_dir()
        print(f"Using default projects directory: {self._projects_dir}")
    
    def _get_default_projects_dir(self):
        """Get default projects directory"""
        # Default to user's Documents/vestim_projects
        docs_dir = os.path.expanduser("~/Documents")
        return os.path.join(docs_dir, "vestim_projects")
    
    def get_projects_directory(self):
        """Get the projects directory path"""
        # Ensure directory exists
        if not os.path.exists(self._projects_dir):
            os.makedirs(self._projects_dir, exist_ok=True)
            print(f"Created projects directory: {self._projects_dir}")
        
        return self._projects_dir
    
    def get_output_directory(self):
        """Get the output directory for jobs (same as projects directory)"""
        return self.get_projects_directory()

# Global instance
_config_manager = None

def get_config_manager():
    """Get the global config manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_projects_directory():
    """Convenience function to get projects directory"""
    return get_config_manager().get_projects_directory()

def get_output_directory():
    """Convenience function to get output directory"""
    return get_config_manager().get_output_directory()
