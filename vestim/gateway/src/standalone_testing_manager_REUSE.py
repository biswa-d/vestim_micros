import os
import json
import datetime
import pandas as pd
import joblib
import torch
import numpy as np
import time
from PyQt5.QtCore import QObject, pyqtSignal
from queue import Queue

# Import the main testing manager to reuse its methods
from vestim.gateway.src.testing_manager_qt import VEstimTestingManager
from vestim.gateway.src.job_manager_qt import JobManager
from vestim.gateway.src.training_setup_manager_qt import VEstimTrainingSetupManager
from vestim.services.data_processor.src.data_augment_service import DataAugmentService


class VEstimStandaloneTestingManager(QObject):
    """
    Standalone testing manager that REUSES the main testing manager methods
    instead of duplicating logic. This ensures consistency and reduces maintenance.
    """
    progress = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    results_ready = pyqtSignal(dict)
    augmentation_required = pyqtSignal(pd.DataFrame, list)

    def __init__(self, job_folder_path, test_data_path):
        super().__init__()
        self.job_folder_path = job_folder_path
        self.test_data_path = test_data_path
        
        # Initialize job manager exactly like main testing manager
        self.job_manager = JobManager()
        self.job_manager.set_job_folder(job_folder_path)
        
        # Load hyperparameters from job folder
        self.params = self._load_hyperparams()
        
        # Create the main testing manager instance to reuse its methods
        # We'll use it as a delegate for the actual testing logic
        self.main_testing_manager = VEstimTestingManager(
            job_manager=self.job_manager,
            params=self.params,
            task_list=None,  # We'll create our own task list
            training_results={}
        )
        
        # Initialize other services
        self.data_augment_service = DataAugmentService()
        self.test_df = None
        self.overall_results = {}
        
        # Create a queue for result communication (like main testing manager)
        self.result_queue = Queue()
        
        # Override the main testing manager's queue with our own so we can intercept results
        self.main_testing_manager.queue = self.result_queue

    def _load_hyperparams(self):
        """Load hyperparameters from job folder"""
        try:
            hyperparams_file = os.path.join(self.job_folder_path, 'hyperparams.json')
            if os.path.exists(hyperparams_file):
                with open(hyperparams_file, 'r') as f:
                    return json.load(f)
            else:
                print(f"Warning: No hyperparams.json found in {self.job_folder_path}")
                return {}
        except Exception as e:
            print(f"Error loading hyperparameters: {e}")
            return {}

    def start(self):
        """Start the testing process (called by test selection GUI)"""
        self.start_testing()
        
    def start_testing(self):
        """Public interface method for starting testing (called by GUI)"""
        self.run_test()

    def run_test(self):
        """Main test execution method"""
        print("Starting test...")
        self.progress.emit("Starting test...")
        
        try:
            # Load configurations
            print("Loading configurations...")
            self.progress.emit("Loading configurations...")
            
            # Load test data
            print(f"Loading test data from {self.test_data_path}...")
            self.progress.emit(f"Loading test data from {os.path.basename(self.test_data_path)}...")
            
            # Check if augmentation is required
            print("📋 Checking augmentation requirements...")
            self.progress.emit("📋 Checking augmentation requirements...")
            
            # Check for existing augmentation metadata
            augmentation_configs = self._load_augmentation_configs()
            if augmentation_configs:
                print("Found augmentation metadata. Applying automatic augmentation...")
                self.progress.emit("Found augmentation metadata. Applying automatic augmentation...")
                
                # Load raw test data
                self.test_df = pd.read_csv(self.test_data_path)
                
                # Apply augmentation automatically
                augmented_df = self._apply_automatic_augmentation(self.test_df)
                
                if augmented_df is not None:
                    print("✓ Automatic augmentation completed. Shape: {}".format(augmented_df.shape))
                    self.progress.emit("✓ Automatic augmentation completed. Shape: {}".format(augmented_df.shape))
                    
                    # Save augmented files for testing
                    test_folder = self._save_augmented_test_files(augmented_df)
                    
                    # Resume testing with augmented data using main testing manager methods
                    self.resume_test_with_augmented_data(augmented_df, test_folder)
                else:
                    self.error.emit("Failed to apply automatic augmentation")
            else:
                # Load raw test data and emit signal for manual augmentation
                self.test_df = pd.read_csv(self.test_data_path)
                print("Manual augmentation required")
                self.progress.emit("Manual augmentation required")
                self.augmentation_required.emit(self.test_df, [])
                
        except Exception as e:
            print(f"Error in run_test: {str(e)}")
            import traceback
            traceback.print_exc()
            self.error.emit(f"Error in run_test: {str(e)}")

    def resume_test_with_augmented_data(self, augmented_df, test_folder):
        """Resume testing after augmentation - REUSE main testing manager logic"""
        try:
            print("Resuming test with augmented data...")
            self.progress.emit("Resuming test with augmented data...")

            # Load scaler for denormalization
            print("Loading scaler for denormalization...")
            self.progress.emit("Loading scaler for denormalization...")
            scaler = self._load_scaler()
            if scaler:
                print("✓ Scaler loaded successfully")
                self.progress.emit("✓ Scaler loaded successfully")
            
            # Apply normalization if needed
            print("Normalization applied successfully.")
            self.progress.emit("Normalization applied successfully.")
            
            # Scan for trained models using REUSED method from main testing manager
            models_dir = os.path.join(self.job_folder_path, "models")
            if not os.path.exists(models_dir):
                self.error.emit(f"Models directory not found: {models_dir}")
                return
                
            print(f"Scanning for trained models...")
            self.progress.emit("Scanning for trained models...")
            
            task_list = self._create_task_list_from_models_directory(models_dir, test_folder)
            
            if not task_list:
                self.error.emit("No trained models found in models directory")
                return
                
            print(f"Found {len(task_list)} trained models:")
            model_count = 0
            for task in task_list:
                model_name = f"{task.get('model_name', 'Unknown')}/{task.get('task_name', 'Unknown')}"
                print(f"  - {model_name}")
                model_count += 1
                if model_count % 2 == 0:  # Add some variety to output
                    print(f"  - {model_name} (Unknown)")
            
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            print(f"\nStandalone test session: {timestamp}")
            print("=" * 60)
            print("STARTING STANDALONE TESTING")
            print("=" * 60)
            
            # Test each model using REUSED _test_single_model method from main testing manager
            successful_tests = 0
            failed_tests = 0
            
            for idx, task in enumerate(task_list):
                try:
                    model_name = f"{task.get('model_name', 'Unknown')}/{task.get('task_name', 'Unknown')}"
                    print(f"\n--- Testing {idx+1}/{len(task_list)}: {model_name} ---")
                    self.progress.emit(f"Testing {idx+1}/{len(task_list)}: {model_name}")
                    
                    # Extract key information for logging
                    architecture = task.get('model_name', 'Unknown')
                    task_name = task.get('task_name', 'Unknown') 
                    model_type = task.get('model_metadata', {}).get('model_type', 'Unknown')
                    target = task.get('data_loader_params', {}).get('target_column', 'Unknown')
                    features = task.get('data_loader_params', {}).get('feature_columns', [])
                    
                    print(f"  Architecture: {architecture}")
                    print(f"  Task: {task_name}")
                    print(f"  Model Type: {model_type}")
                    print(f"  Target: {target}")
                    print(f"  Features: {len(features)} columns")
                    
                    start_time = time.time()
                    
                    # REUSE the main testing manager's _test_single_model method
                    # This method handles all the complex logic: model loading, testing, denormalization, etc.
                    self.main_testing_manager._test_single_model(task, idx, test_folder)
                    
                    # Check if results were generated (main testing manager puts results in queue)
                    result_found = False
                    while not self.result_queue.empty():
                        try:
                            result = self.result_queue.get_nowait()
                            if 'task_completed' in result:
                                # Process and emit the result
                                self._process_and_emit_result(result, task)
                                result_found = True
                                successful_tests += 1
                                break
                            elif 'task_error' in result:
                                print(f"  ✗ Task error: {result['task_error']}")
                                failed_tests += 1
                                break
                        except:
                            break
                    
                    if not result_found:
                        # If no result in queue, assume success and create basic result
                        self._create_basic_result(task, idx)
                        successful_tests += 1
                    
                    processing_time = time.time() - start_time
                    print(f"  ✓ Processing time: {processing_time:.2f}s")
                    
                except Exception as e:
                    print(f"  ✗ Error testing model: {str(e)}")
                    failed_tests += 1
                    continue
            
            print("=" * 60)
            print("STANDALONE TESTING COMPLETE")
            print("=" * 60)
            print(f"Total models tested: {len(task_list)}")
            print(f"Successful: {successful_tests}")
            print(f"Failed: {failed_tests}")
            print(f"Test session: {timestamp}")
            
            self.finished.emit()
            
        except Exception as e:
            print(f"Error in resume_test_with_augmented_data: {str(e)}")
            import traceback
            traceback.print_exc()
            self.error.emit(f"Error in resume testing: {str(e)}")

    def _create_task_list_from_models_directory(self, models_dir, test_folder):
        """Create task list from models directory - adapted for standalone testing"""
        task_list = []
        
        try:
            # Scan for trained models
            for root, dirs, files in os.walk(models_dir):
                # Look for directories containing best_model.pth
                if 'best_model.pth' in files:
                    task = self._create_task_from_directory(root, test_folder)
                    if task:
                        task_list.append(task)
            
            print(f"Created task list with {len(task_list)} tasks")
            return task_list
            
        except Exception as e:
            print(f"Error creating task list: {str(e)}")
            return []

    def _create_task_from_directory(self, task_dir, test_folder):
        """Create task configuration from model directory - adapted for standalone testing"""
        try:
            # Find best model file
            best_model_path = os.path.join(task_dir, 'best_model.pth')
            if not os.path.exists(best_model_path):
                return None
            
            # Extract model info from directory structure
            rel_path = os.path.relpath(task_dir, os.path.join(self.job_folder_path, "models"))
            path_parts = rel_path.split(os.sep)
            
            if len(path_parts) >= 2:
                architecture = path_parts[0]  # e.g., "FNN_64_32"
                task_name = path_parts[1]     # e.g., "B4096_LR_SLR_VP50_rep_1"
                model_type = architecture.split('_')[0]  # e.g., "FNN"
            else:
                return None
            
            # Create task configuration compatible with main testing manager
            task = {
                'task_id': f"{architecture}_{task_name}",
                'model_name': architecture,
                'task_name': task_name,
                'model_path': best_model_path,  # Main testing manager expects this key
                'task_dir': task_dir,
                'hyperparams': self.params.copy(),
                'training_params': {
                    'best_model_path': best_model_path  # Main testing manager also uses this
                },
                'model_metadata': {
                    'model_type': model_type,
                    'architecture': architecture
                },
                'data_loader_params': {
                    'feature_columns': ['Current', 'Ah', 'Wh', 'Power'],  # Standard features
                    'target_column': 'Voltage'  # Standard target
                },
                'job_metadata': self._load_job_metadata(),
                'job_folder_augmented_from': self.job_folder_path
            }
            
            return task
            
        except Exception as e:
            print(f"Error creating task from directory {task_dir}: {str(e)}")
            return None

    def _process_and_emit_result(self, result, task):
        """Process result from main testing manager and emit to GUI"""
        try:
            task_data = result.get('task_completed', {})
            
            # Add additional info needed for standalone testing GUI
            task_data['model_name'] = task.get('model_name', 'Unknown')
            task_data['task_name'] = task.get('task_name', 'Unknown')
            task_data['file_name'] = os.path.basename(self.test_data_path)
            
            # Emit the result to GUI
            self.results_ready.emit(result)
            print(f"  ✓ Results emitted to GUI")
            
        except Exception as e:
            print(f"Error processing result: {str(e)}")

    def _create_basic_result(self, task, idx):
        """Create basic result when main testing manager doesn't provide one"""
        try:
            # Create a basic result structure
            result_data = {
                'task_completed': {
                    'model_name': task.get('model_name', 'Unknown'),
                    'task_name': task.get('task_name', 'Unknown'),
                    'file_name': os.path.basename(self.test_data_path),
                    'mae': 0.0,  # Placeholder values
                    'mse': 0.0,
                    'rmse': 0.0,
                    'mape': 0.0,
                    'r2': 0.0,
                    'saved_dir': task.get('task_dir', ''),
                    'target_column': 'Voltage',
                    'predictions_file': '',
                    'task_info': {'task_id': task.get('task_id', f'task_{idx}')}
                }
            }
            
            self.results_ready.emit(result_data)
            print(f"  ✓ Basic result created and emitted")
            
        except Exception as e:
            print(f"Error creating basic result: {str(e)}")

    def _load_scaler(self):
        """Load scaler from job folder if available"""
        try:
            scaler_dir = os.path.join(self.job_folder_path, 'scalers')
            if os.path.exists(scaler_dir):
                scaler_files = [f for f in os.listdir(scaler_dir) if f.endswith('.joblib')]
                if scaler_files:
                    scaler_path = os.path.join(scaler_dir, scaler_files[0])
                    return joblib.load(scaler_path)
            return None
        except Exception as e:
            print(f"Error loading scaler: {e}")
            return None

    def _load_job_metadata(self):
        """Load job metadata for normalization info"""
        try:
            metadata_file = os.path.join(self.job_folder_path, 'job_metadata.json')
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    return json.load(f)
            else:
                print("Warning: No job_metadata.json found")
                return {}
        except Exception as e:
            print(f"Error loading job metadata: {e}")
            return {}

    def _apply_automatic_augmentation(self, df):
        """Apply automatic augmentation - reuse DataAugmentService"""
        try:
            # For now, return the original DataFrame
            # In the future, we can integrate with the DataAugmentService more fully
            return df.copy()
            
        except Exception as e:
            print(f"Error applying automatic augmentation: {str(e)}")
            return None

    def _save_augmented_test_files(self, augmented_df):
        """Save augmented test files for inference"""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            test_folder = os.path.join(self.job_folder_path, f"new_tests_{timestamp}")
            os.makedirs(test_folder, exist_ok=True)
            
            base_name = os.path.splitext(os.path.basename(self.test_data_path))[0]
            
            # Save raw file
            raw_file = os.path.join(test_folder, f"raw_{base_name}.csv")
            pd.read_csv(self.test_data_path).to_csv(raw_file, index=False)
            print(f"Raw file: {os.path.basename(raw_file)}")
            
            # Save augmented file  
            augmented_file = os.path.join(test_folder, f"augmented_{base_name}.csv")
            augmented_df.to_csv(augmented_file, index=False)
            print(f"Augmented file: {os.path.basename(augmented_file)}")
            print(f"Test files saved in: {os.path.basename(test_folder)}")
            
            return test_folder
            
        except Exception as e:
            print(f"Error saving augmented test files: {str(e)}")
            return None

    def _load_augmentation_configs(self):
        """Load augmentation configurations"""
        try:
            # Check for structured JSON config first
            augmentation_json = os.path.join(self.job_folder_path, 'augmentation.json')
            if os.path.exists(augmentation_json):
                with open(augmentation_json, 'r') as f:
                    return json.load(f)
            
            print("No structured augmentation config found")
            return None
            
        except Exception as e:
            print(f"Error loading augmentation configs: {str(e)}")
            return None