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
        self._data_dir = None
        self._defaults_dir = None
        self._default_settings = {}
        self._load_config()
        self._load_default_settings()
    
    def _load_config(self):
        """
        Load configuration for Vestim.
        Priority order:
        1. VESTIM_PROJECT_DIR environment variable (set by launcher).
        2. vestim_config.json file (for direct execution or fallback).
        3. Default directories (as a last resort).
        """
        # 1. Check for environment variable from launcher
        project_dir_env = os.environ.get('VESTIM_PROJECT_DIR')
        if project_dir_env and Path(project_dir_env).exists():
            self._projects_dir = project_dir_env
            self._data_dir = str(Path(project_dir_env) / 'data')
            self._defaults_dir = str(Path(project_dir_env) / 'defaults_templates')
            print(f"Loaded configuration from VESTIM_PROJECT_DIR: {self._projects_dir}")
            return

        # 2. Check for vestim_config.json
        try:
            if getattr(sys, 'frozen', False):
                app_root = Path(sys.executable).parent
            else:
                app_root = Path(__file__).parent.parent

            config_path = app_root / "vestim_config.json"

            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                projects_dir = config.get('projects_directory')
                if projects_dir and Path(projects_dir).exists():
                    self._projects_dir = projects_dir
                    self._data_dir = config.get('data_directory')
                    self._defaults_dir = config.get('defaults_directory')
                    print(f"Loaded configuration from {config_path}")
                    return
        except Exception as e:
            print(f"Error loading vestim_config.json: {e}")

        # 3. Fallback to default locations
        print("Using default directory configuration.")
        self._projects_dir = self._get_default_projects_dir()
        self._data_dir = self._get_default_data_dir()
        self._defaults_dir = self._get_default_defaults_dir()
        
        # Only show message for compiled executable, be quiet during development
        if getattr(sys, 'frozen', False):
            print(f"Using default projects directory: {self._projects_dir}")
            print(f"Using default data directory: {self._data_dir}")
            print(f"Using default hyperparams templates directory: {self._defaults_dir}")
        # For development, just use default silently
    
    def _get_default_projects_dir(self):
        """Get default projects directory"""
        if getattr(sys, 'frozen', False):
            # Running as compiled executable - use user's Documents folder
            docs_dir = os.path.expanduser("~/Documents")
            return os.path.join(docs_dir, "vestim_projects")
        else:
            # Running as Python script - use repository's output directory
            script_dir = Path(__file__).parent  # vestim/
            repo_root = script_dir.parent  # repo root
            return str(repo_root / "output")
    
    def _get_default_data_dir(self):
        """Get default data directory"""
        if getattr(sys, 'frozen', False):
            # Running as compiled executable - use configured projects folder/data
            # This should prefer the installed config over hardcoded paths
            if self._projects_dir:
                return os.path.join(self._projects_dir, "data")
            else:
                # Only fallback to default if no config was loaded
                projects_dir = self._get_default_projects_dir()
                return os.path.join(projects_dir, "data")
        else:
            # Running as Python script - use repository's data directory
            script_dir = Path(__file__).parent  # vestim/
            repo_root = script_dir.parent  # repo root
            return str(repo_root / "data")
    
    def _get_default_defaults_dir(self):
        """Get default hyperparams templates directory"""
        if getattr(sys, 'frozen', False):
            # Running as compiled executable - use projects folder/default_hyperparams
            projects_dir = self._get_default_projects_dir()
            return os.path.join(projects_dir, "default_hyperparams")
        else:
            # Running as Python script - use repository's defaults_templates directory
            script_dir = Path(__file__).parent  # vestim/
            repo_root = script_dir.parent  # repo root
            return str(repo_root / "defaults_templates")
    
    def get_projects_directory(self):
        """Get the projects directory path"""
        # Ensure directory exists
        if not os.path.exists(self._projects_dir):
            os.makedirs(self._projects_dir, exist_ok=True)
            # Only show message for compiled executable
            if getattr(sys, 'frozen', False):
                print(f"Created projects directory: {self._projects_dir}")
        
        return self._projects_dir
    
    def get_data_directory(self):
        """Get the default data directory path"""
        # Ensure directory exists
        if not os.path.exists(self._data_dir):
            os.makedirs(self._data_dir, exist_ok=True)
            # Create initial data structure with demo files
            self.create_initial_data_structure()
            # Only show message for compiled executable
            if getattr(sys, 'frozen', False):
                print(f"Created data directory with demo files: {self._data_dir}")
        else:
            # Check if subdirectories exist, create them if not
            for subdir in ["train_data", "val_data", "test_data"]:
                subdir_path = os.path.join(self._data_dir, subdir)
                if not os.path.exists(subdir_path):
                    os.makedirs(subdir_path, exist_ok=True)
        
        return self._data_dir
    
    def get_defaults_directory(self):
        """Get the default hyperparams templates directory path"""
        # Ensure directory exists
        if not os.path.exists(self._defaults_dir):
            os.makedirs(self._defaults_dir, exist_ok=True)
            # Create initial defaults structure with template files
            self.create_initial_defaults_structure()
            # Only show message for compiled executable
            if getattr(sys, 'frozen', False):
                print(f"Created default hyperparams directory with template files: {self._defaults_dir}")
        
        return self._defaults_dir
    
    def get_output_directory(self):
        """Get the output directory for jobs (same as projects directory)"""
        return self.get_projects_directory()
    
    def _load_default_settings(self):
        """Load default settings from configuration file"""
        try:
            # Look for default settings file
            if getattr(sys, 'frozen', False):
                # Running as compiled executable - check projects directory first
                projects_dir = self.get_projects_directory()
                settings_path = Path(projects_dir) / "default_settings.json"
                
                if not settings_path.exists():
                    # Fallback to application directory
                    app_dir = Path(sys.executable).parent
                    settings_path = app_dir / "default_settings.json"
            else:
                # Running as script - look in script directory
                app_dir = Path(__file__).parent
                settings_path = app_dir / "default_settings.json"
            
            if settings_path.exists():
                with open(settings_path, 'r') as f:
                    self._default_settings = json.load(f)
                    if getattr(sys, 'frozen', False):
                        print(f"Loaded default settings from: {settings_path}")
            else:
                # Create default settings with fallback folder paths
                if getattr(sys, 'frozen', False):
                    print(f"Default settings not found at: {settings_path}")
                    print(f"Creating fallback settings using data directory: {self.get_data_directory()}")
                self._default_settings = self._get_initial_default_settings()
                self._save_default_settings()
                
        except Exception as e:
            print(f"Could not load default settings: {e}")
            self._default_settings = self._get_initial_default_settings()
    
    def _get_initial_default_settings(self):
        """Get initial default settings for first-time setup"""
        data_dir = self.get_data_directory()
        return {
            "last_used": {
                "train_folder": os.path.join(data_dir, "train_data"),
                "val_folder": os.path.join(data_dir, "val_data"), 
                "test_folder": os.path.join(data_dir, "test_data"),
                "file_format": "csv"
            },
            "default_folders": {
                "train_folder": os.path.join(data_dir, "train_data"),
                "val_folder": os.path.join(data_dir, "val_data"),
                "test_folder": os.path.join(data_dir, "test_data")
            }
        }
    
    def _save_default_settings(self):
        """Save default settings to configuration file"""
        try:
            # Determine where to save the settings
            if getattr(sys, 'frozen', False):
                # Running as compiled executable - save in projects directory
                projects_dir = self.get_projects_directory()
                settings_path = Path(projects_dir) / "default_settings.json"
            else:
                # Running as script - save in script directory
                app_dir = Path(__file__).parent
                settings_path = app_dir / "default_settings.json"
            
            with open(settings_path, 'w') as f:
                json.dump(self._default_settings, f, indent=4)
                
        except Exception as e:
            print(f"Could not save default settings: {e}")
    
    def get_default_folders(self):
        """Get default folder paths for train/val/test data"""
        return self._default_settings.get("last_used", {})
    
    def update_last_used_folders(self, train_folder=None, val_folder=None, test_folder=None, file_format=None):
        """Update the last used folder paths and file format"""
        if "last_used" not in self._default_settings:
            self._default_settings["last_used"] = {}
            
        if train_folder:
            self._default_settings["last_used"]["train_folder"] = train_folder
        if val_folder:
            self._default_settings["last_used"]["val_folder"] = val_folder
        if test_folder:
            self._default_settings["last_used"]["test_folder"] = test_folder
        if file_format:
            self._default_settings["last_used"]["file_format"] = file_format
            
        self._save_default_settings()
    
    def get_default_file_format(self):
        """Get the last used file format"""
        return self._default_settings.get("last_used", {}).get("file_format", "csv")
    
    def update_last_used_hyperparams(self, hyperparams):
        """Update the last used hyperparameters in hyperparams_last_used.json."""
        try:
            # Always save in defaults_templates folder in project directory
            repo_root = Path(__file__).parent.parent
            defaults_dir = repo_root / "defaults_templates"
            defaults_dir.mkdir(exist_ok=True)
            filepath = defaults_dir / "hyperparams_last_used.json"
            with open(filepath, 'w') as f:
                json.dump(hyperparams, f, indent=4)
            print(f"[VEstim] Saved last used hyperparams to: {filepath}")
        except Exception as e:
            print(f"Could not save last used hyperparams: {e}")

    def get_default_hyperparams(self):
        """Get default hyperparameters - from last used file or system defaults."""
        try:
            # Always load from defaults_templates folder in project directory
            repo_root = Path(__file__).parent.parent
            defaults_dir = repo_root / "defaults_templates"
            filepath = defaults_dir / "hyperparams_last_used.json"
            print(f"[VEstim] Attempting to load last used hyperparams from: {filepath}")
            if filepath.exists():
                print(f"[VEstim] Found last used hyperparams file: {filepath}")
                with open(filepath, 'r') as f:
                    return json.load(f)
            else:
                print(f"[VEstim] WARNING: last used hyperparams file not found at {filepath}. Falling back to initial defaults.")
        except Exception as e:
            print(f"[VEstim] ERROR: Could not load last used hyperparams: {e}. Falling back to initial defaults.")

        # Fallback to initial defaults
        return self._get_initial_default_hyperparams()
    
    def _get_initial_default_hyperparams(self):
        """Get initial default hyperparameters for first-time setup"""
        return {
            "FEATURE_COLUMNS": ["SOC", "Current", "Temp"],  # Updated to match actual CSV columns
            "TARGET_COLUMN": "Voltage",
            "MODEL_TYPE": "LSTM",
            "LAYERS": "1",
            "HIDDEN_UNITS": "10",
            "TRAINING_METHOD": "Sequence-to-Sequence",
            "LOOKBACK": "400",
            "BATCH_TRAINING": True,
            "BATCH_SIZE": "200",
            "SCHEDULER_TYPE": "StepLR",
            "INITIAL_LR": "0.0001",
            "LR_PARAM": "0.1",
            "LR_PERIOD": "2",
            "PLATEAU_PATIENCE": "10",
            "PLATEAU_FACTOR": "0.1",
            "VALID_PATIENCE": "10",
            "VALID_FREQUENCY": "1",
            "MAX_EPOCHS": "5",
            "REPETITIONS": 1,
            "DEVICE_SELECTION": "CPU",
            "MAX_TRAINING_TIME_SECONDS": 0,
            "TRAIN_VAL_SPLIT": "0.8",
            "LR_DROP_FACTOR": "0.5",
            "LR_DROP_PERIOD": "1",
            "ValidFrequency": "1",
            "SEQUENCE_SPLIT_METHOD": "temporal",
            "MAX_TRAIN_HOURS": "0",
            "MAX_TRAIN_MINUTES": "30",
            "MAX_TRAIN_SECONDS": "0"
        }
    
    def load_hyperparams_from_root(self):
        """Load hyperparameters from the root hyperparams.json file if it exists"""
        try:
            # Look for hyperparams.json in application root directory
            if getattr(sys, 'frozen', False):
                # Running as compiled executable
                app_dir = Path(sys.executable).parent
            else:
                # Running as script - look in repository root
                script_dir = Path(__file__).parent  # vestim/
                app_dir = script_dir.parent  # repo root
            
            hyperparams_path = app_dir / "hyperparams.json"
            
            if hyperparams_path.exists():
                with open(hyperparams_path, 'r') as f:
                    return json.load(f)
            else:
                return None
                
        except Exception as e:
            print(f"Could not load root hyperparams.json: {e}")
            return None
    
    def create_initial_data_structure(self):
        """Create initial data directory structure with demo files for first-time setup"""
        data_dir = self.get_data_directory()
        
        # Create the subdirectories
        train_dir = os.path.join(data_dir, "train_data")
        val_dir = os.path.join(data_dir, "val_data") 
        test_dir = os.path.join(data_dir, "test_data")
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # For compiled executable, copy demo files from installer assets
        if getattr(sys, 'frozen', False):
            self._copy_demo_files_from_assets(data_dir)
        else:
            # During development, copy from existing data folder if available
            self._copy_demo_files_from_dev_data(data_dir)
        
        return data_dir
    
    def create_initial_defaults_structure(self):
        """Create initial defaults directory structure with template files for first-time setup"""
        defaults_dir = self.get_defaults_directory()
        
        # For compiled executable, copy template files from installer assets
        if getattr(sys, 'frozen', False):
            self._copy_defaults_from_assets(defaults_dir)
        else:
            # During development, copy from existing defaults_templates folder if available
            self._copy_defaults_from_dev_templates(defaults_dir)
        
        return defaults_dir
    
    def _copy_demo_files_from_assets(self, data_dir):
        """Copy demo files from installer assets to data directory"""
        try:
            # Get the application directory where assets should be bundled
            app_dir = Path(sys.executable).parent
            assets_dir = app_dir / "installer_assets" / "demo_data"
            
            if assets_dir.exists():
                import shutil
                
                # Copy each subdirectory
                for subdir in ["train_data", "val_data", "test_data"]:
                    source_dir = assets_dir / subdir
                    target_dir = Path(data_dir) / subdir
                    
                    if source_dir.exists():
                        # Copy all files from source to target
                        for file_path in source_dir.glob("*"):
                            if file_path.is_file():
                                shutil.copy2(file_path, target_dir)
                                print(f"Copied demo file: {file_path.name} to {subdir}")
                                
                # Also copy the user README to projects directory
                readme_source = app_dir / "USER_README.md"
                if readme_source.exists():
                    projects_dir = self.get_projects_directory()
                    readme_target = Path(projects_dir) / "README.md"
                    shutil.copy2(readme_source, readme_target)
                    print(f"Copied user README to projects directory")
                    
        except Exception as e:
            print(f"Could not copy demo files from assets: {e}")
    
    def _copy_demo_files_from_dev_data(self, data_dir):
        """Copy demo files from development data folder (for development mode)"""
        try:
            # Get the repository root data directory
            script_dir = Path(__file__).parent  # vestim/
            repo_root = script_dir.parent  # repo root
            source_data_dir = repo_root / "data"
            
            if source_data_dir.exists():
                import shutil
                
                # Copy training data
                train_source = source_data_dir / "train_data" / "Combined_Training31-Aug-2023.csv"
                if train_source.exists():
                    train_target = Path(data_dir) / "train_data" / "demo_train_data.csv"
                    shutil.copy2(train_source, train_target)
                    print(f"Copied demo training data to {train_target}")
                
                # Copy validation data
                val_source = source_data_dir / "val_data" / "raw_data" / "105_UDDS_n20C.csv"
                if val_source.exists():
                    val_target = Path(data_dir) / "val_data" / "demo_validation_data.csv"
                    shutil.copy2(val_source, val_target)
                    print(f"Copied demo validation data to {val_target}")
                
                # Copy test data
                test_source = source_data_dir / "test_data" / "raw_data" / "107_LA92_n20C.csv"
                if test_source.exists():
                    test_target = Path(data_dir) / "test_data" / "demo_test_data.csv"
                    shutil.copy2(test_source, test_target)
                    print(f"Copied demo test data to {test_target}")
                    
                # Also copy the combined testing file as an alternative
                combined_test_source = source_data_dir / "Combined_Testing31-Aug-2023.csv"
                if combined_test_source.exists():
                    combined_test_target = Path(data_dir) / "test_data" / "demo_combined_test_data.csv"
                    shutil.copy2(combined_test_source, combined_test_target)
                    print(f"Copied demo combined test data to {combined_test_target}")
                    
        except Exception as e:
            print(f"Could not copy demo files from development data: {e}")
    
    def _copy_defaults_from_assets(self, defaults_dir):
        """Copy default template files from installer assets to defaults directory"""
        try:
            # Get the application directory where assets should be bundled
            app_dir = Path(sys.executable).parent
            assets_dir = app_dir / "installer_assets" / "default_hyperparams"
            
            if assets_dir.exists():
                import shutil
                
                # Copy all template files
                for file_path in assets_dir.glob("*.json"):
                    if file_path.is_file():
                        target_path = Path(defaults_dir) / file_path.name
                        shutil.copy2(file_path, target_path)
                        print(f"Copied hyperparams template file: {file_path.name}")
                        
        except Exception as e:
            print(f"Could not copy hyperparams template files from assets: {e}")
    
    def _copy_defaults_from_dev_templates(self, defaults_dir):
        """Copy template files from development defaults_templates folder (for development mode)"""
        try:
            # Get the repository root defaults_templates directory
            script_dir = Path(__file__).parent  # vestim/
            repo_root = script_dir.parent  # repo root
            source_templates_dir = repo_root / "defaults_templates"
            
            if source_templates_dir.exists():
                import shutil
                
                # Copy all JSON template files
                for file_path in source_templates_dir.glob("*.json"):
                    if file_path.is_file():
                        target_path = Path(defaults_dir) / file_path.name
                        shutil.copy2(file_path, target_path)
                        print(f"Copied hyperparams template file: {file_path.name}")
                        
        except Exception as e:
            print(f"Could not copy hyperparams template files from development templates: {e}")

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

def get_data_directory():
    """Convenience function to get default data directory"""
    return get_config_manager().get_data_directory()

def get_defaults_directory():
    """Convenience function to get default hyperparams templates directory"""
    return get_config_manager().get_defaults_directory()

def get_output_directory():
    """Convenience function to get output directory"""
    return get_config_manager().get_output_directory()

def get_default_data_directory():
    """Convenience function to get default data directory (alias for get_data_directory)"""
    return get_config_manager().get_data_directory()

def get_default_folders():
    """Convenience function to get default folder paths"""
    return get_config_manager().get_default_folders()

def update_last_used_folders(train_folder=None, val_folder=None, test_folder=None, file_format=None):
    """Convenience function to update last used folder paths"""
    return get_config_manager().update_last_used_folders(train_folder, val_folder, test_folder, file_format)

def get_default_file_format():
    """Convenience function to get default file format"""
    return get_config_manager().get_default_file_format()

def update_last_used_hyperparams(hyperparams):
    """Convenience function to update last used hyperparameters"""
    return get_config_manager().update_last_used_hyperparams(hyperparams)

def get_default_hyperparams():
    """Convenience function to get default hyperparameters"""
    return get_config_manager().get_default_hyperparams()

def load_hyperparams_from_root():
    """Convenience function to load hyperparameters from root hyperparams.json"""
    return get_config_manager().load_hyperparams_from_root()
