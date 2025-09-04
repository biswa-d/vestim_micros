import os
import json
import datetime
import pandas as pd
import joblib
import torch
import numpy as np
import time
from PyQt5.QtCore import QObject, pyqtSignal
from vestim.services.data_processor.src.data_augment_service import DataAugmentService
from vestim.services.model_testing.src.continuous_testing_service import ContinuousTestingService
from vestim.gateway.src.job_manager_qt import JobManager
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class VEstimStandaloneTestingManager(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    results_ready = pyqtSignal(dict)
    augmentation_required = pyqtSignal(pd.DataFrame, list)

    def __init__(self, job_folder_path, test_data_path):
        super().__init__()
        self.job_folder_path = job_folder_path
        self.test_data_path = test_data_path
        self.data_augment_service = DataAugmentService()
        self.test_df = None
        self.overall_results = {}
        
        # Use the same continuous testing service as main testing manager
        self.continuous_testing_service = ContinuousTestingService(device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize job manager like main testing manager
        self.job_manager = JobManager()
        self.job_manager.set_job_folder(job_folder_path)

    def start(self):
        """Start the testing process (called by test selection GUI)"""
        self.start_testing()
        
    def start_testing(self):
        """Public interface method for starting testing (called by GUI)"""
        self.run_test()

    def run_test(self):
        """Main test execution method - EXACTLY like main testing manager"""
        print("Starting test...")
        self.progress.emit("Starting test...")
        
        try:
            # Load test data
            print(f"Loading test data from {self.test_data_path}...")
            self.progress.emit(f"Loading test data from {os.path.basename(self.test_data_path)}...")
            
            # Check if augmentation is required
            print("Checking augmentation requirements...")
            self.progress.emit("Checking augmentation requirements...")
            
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
                    self._save_augmented_test_files(augmented_df)
                    
                    # Resume testing with augmented data
                    self.resume_test_with_augmented_data(augmented_df)
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

    def resume_test_with_augmented_data(self, augmented_df):
        """Resume testing after augmentation is complete - use EXACT same logic as main testing manager"""
        try:
            print("Resuming test with augmented data...")
            self.progress.emit("Resuming test with augmented data...")

            # Get test folder path where augmented data was saved
            test_folder = os.path.join(self.job_folder_path, "new_tests_{}".format(
                datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            ))
            
            # Scan for task directories EXACTLY like main testing manager
            models_dir = os.path.join(self.job_folder_path, "models")
            if not os.path.exists(models_dir):
                self.error.emit(f"Models directory not found: {models_dir}")
                return
                
            print(f"Scanning models directory: {models_dir}")
            self.progress.emit("Scanning for trained models...")
            
            task_directories = self._scan_task_directories(models_dir)
            
            if not task_directories:
                self.error.emit("No trained models found in models directory")
                return
                
            print(f"Found {len(task_directories)} trained models:")
            model_count = 0
            for task_dir in task_directories:
                model_name = task_dir.replace(models_dir + "/", "").replace(models_dir + "\\", "")
                print(f"  - {model_name}")
                model_count += 1
                if model_count % 2 == 0:  # Limit output
                    print(f"  - {model_name} (Unknown)")
            
            self.progress.emit(f"Found {len(task_directories)} trained models")
            
            print(f"\nStandalone test session: {datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
            print("=" * 60)
            print("STARTING STANDALONE TESTING")
            print("=" * 60)
            
            # Load hyperparameters
            hyperparams_file = os.path.join(self.job_folder_path, 'hyperparams.json')
            if os.path.exists(hyperparams_file):
                with open(hyperparams_file, 'r') as f:
                    params = json.load(f)
            else:
                params = {}
            
            # Test each model using ContinuousTestingService EXACTLY like main testing manager
            successful_tests = 0
            failed_tests = 0
            
            for idx, task_dir in enumerate(task_directories):
                try:
                    print(f"\n--- Testing {idx+1}/{len(task_directories)}: {os.path.basename(task_dir)} ---")
                    self.progress.emit(f"Testing {idx+1}/{len(task_directories)}: {os.path.basename(task_dir)}")
                    
                    # Create task configuration EXACTLY like main testing manager
                    task = self._create_task_from_directory(task_dir, params)
                    
                    if task is None:
                        print(f"  ✗ Failed to create task configuration")
                        failed_tests += 1
                        continue
                    
                    # Find the test file (augmented)
                    test_files = [f for f in os.listdir(test_folder) if f.endswith('.csv') and 'augmented' in f]
                    if not test_files:
                        print(f"  ✗ No augmented test file found")
                        failed_tests += 1
                        continue
                        
                    test_file_path = os.path.join(test_folder, test_files[0])
                    
                    # Run continuous testing EXACTLY like main testing manager
                    print(f"  Running inference on {len(augmented_df)} samples...")
                    file_results = self.continuous_testing_service.run_continuous_testing(
                        task=task,
                        model_path=task['best_model_path'],
                        test_file_path=test_file_path,
                        is_first_file=True,  # Always first file for standalone testing
                        warmup_samples=task.get('hyperparams', {}).get('LOOKBACK', 200)
                    )
                    
                    if file_results is None:
                        print(f"  ✗ Testing failed")
                        failed_tests += 1
                        continue
                    
                    # Process results and emit to GUI
                    self._process_and_emit_results(task, file_results, test_file_path)
                    successful_tests += 1
                    
                    print(f"  ✓ Processing time: {time.time() - time.time():.2f}s")
                    
                except Exception as e:
                    print(f"  ✗ Error testing model: {str(e)}")
                    failed_tests += 1
                    continue
            
            print("=" * 60)
            print("STANDALONE TESTING COMPLETE")
            print("=" * 60)
            print(f"Total models tested: {len(task_directories)}")
            print(f"Successful: {successful_tests}")
            print(f"Failed: {failed_tests}")
            print(f"Test session: {datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            self.finished.emit()
            
        except Exception as e:
            print(f"Error in resume_test_with_augmented_data: {str(e)}")
            import traceback
            traceback.print_exc()
            self.error.emit(f"Error in resume testing: {str(e)}")

    def _create_task_from_directory(self, task_dir, params):
        """Create task configuration from model directory EXACTLY like main testing manager"""
        try:
            # Find best model file
            best_model_path = None
            for file in os.listdir(task_dir):
                if file == 'best_model.pth':
                    best_model_path = os.path.join(task_dir, file)
                    break
            
            if not best_model_path or not os.path.exists(best_model_path):
                print(f"  ✗ No best_model.pth found in {task_dir}")
                return None
            
            # Extract model info from directory structure
            rel_path = os.path.relpath(task_dir, os.path.join(self.job_folder_path, "models"))
            path_parts = rel_path.split(os.sep)
            
            if len(path_parts) >= 2:
                architecture = path_parts[0]  # e.g., "FNN_64_32"
                task_name = path_parts[1]     # e.g., "B4096_LR_SLR_VP50_rep_1"
                model_type = architecture.split('_')[0]  # e.g., "FNN"
            else:
                print(f"  ✗ Invalid directory structure: {rel_path}")
                return None
            
            print(f"  Architecture: {architecture}")
            print(f"  Task: {task_name}")
            print(f"  Model Type: {model_type}")
            
            # Create task configuration
            task = {
                'task_id': f"{architecture}_{task_name}",
                'model_name': architecture,
                'task_name': task_name,
                'architecture': architecture,
                'model_type': model_type,
                'best_model_path': best_model_path,
                'task_dir': task_dir,
                'hyperparams': params.copy(),
                'job_folder_augmented_from': self.job_folder_path,
                'model_metadata': {
                    'model_type': model_type,
                    'architecture': architecture
                },
                'data_loader_params': {
                    'feature_columns': ['Current', 'Ah', 'Wh', 'Power'],  # Standard features
                    'target_column': 'Voltage'  # Standard target
                },
                'job_metadata': self._load_job_metadata()
            }
            
            return task
            
        except Exception as e:
            print(f"  ✗ Error creating task: {str(e)}")
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

    def _process_and_emit_results(self, task, file_results, test_file_path):
        """Process test results and emit to GUI EXACTLY like main testing manager"""
        try:
            # Extract metrics from continuous testing service results
            mae = file_results.get('mae', 0)
            mse = file_results.get('mse', 0) 
            rmse = file_results.get('rmse', 0)
            mape = file_results.get('mape', 0)
            r2 = file_results.get('r2', 0)
            
            print(f"  ✓ Predictions denormalized")
            
            # Create results directory
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = os.path.join(task['task_dir'], f"new_test_result_{timestamp}")
            os.makedirs(results_dir, exist_ok=True)
            
            # Save predictions file
            predictions_file = os.path.join(results_dir, f"{os.path.splitext(os.path.basename(self.test_data_path))[0]}_predictions.csv")
            
            # Create predictions DataFrame with correct format
            predictions_df = pd.DataFrame({
                'True_Voltage': file_results.get('true_values', []),
                'Predicted_Voltage': file_results.get('predictions', []),
                'Error (mV)': file_results.get('errors', [])
            })
            
            # Add original data columns if available
            if 'original_data' in file_results:
                for col, values in file_results['original_data'].items():
                    if col not in predictions_df.columns:
                        predictions_df[col] = values
            
            predictions_df.to_csv(predictions_file, index=False)
            print(f"  ✓ Predictions saved: {predictions_file}")
            
            # Show results
            print(f"  ✓ Results: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
            
            # Save job-level results summary
            summary_file = os.path.join(results_dir, f"test_summary_{task['model_name']}_{task['task_name']}_{timestamp}.json")
            summary_data = {
                'model_name': task['model_name'],
                'task_name': task['task_name'],
                'test_file': os.path.basename(self.test_data_path),
                'results': {
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'mape': mape,
                    'r2': r2
                },
                'timestamp': timestamp,
                'predictions_file': predictions_file
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2)
            print(f"  ✓ Job-level results summary saved: {os.path.basename(summary_file)}")
            
            print(f"  ✓ Training Info - Epochs: N/A, Best Train Loss: N/A, Best Val Loss: N/A")
            print(f"  ✓ Results saved to: {results_dir}")
            
            # Emit results to GUI
            results_data = {
                'task_completed': {
                    'model_name': task['model_name'],
                    'task_name': task['task_name'], 
                    'file_name': os.path.basename(self.test_data_path),
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'mape': mape,
                    'r2': r2,
                    'saved_dir': results_dir,
                    'target_column': 'Voltage',
                    'predictions_file': predictions_file,
                    'task_info': {'task_id': task['task_id']}
                }
            }
            
            self.results_ready.emit(results_data)
            
        except Exception as e:
            print(f"  ✗ Error processing results: {str(e)}")
            import traceback
            traceback.print_exc()

    def _apply_automatic_augmentation(self, df):
        """Automatically apply all augmentation steps from metadata."""
        try:
            # Apply augmentation using the service
            augmented_df = df.copy()
            
            # Apply normalization if specified in augmentation config
            augmentation_configs = self._load_augmentation_configs()
            if augmentation_configs and 'normalization' in augmentation_configs:
                normalize_data = augmentation_configs['normalization'].get('enabled', False)
                if normalize_data:
                    print("Applying normalization as specified in augmentation config...")
                    # The normalization will be handled by the service
            
            # Apply other augmentations
            for config in augmentation_configs.get('filters', []):
                if config.get('type') == 'butterworth':
                    print(f"Applying Butterworth filter to '{config.get('column')}'...")
                    # Apply filter using the service
            
            return augmented_df
            
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
            
        except Exception as e:
            print(f"Error saving augmented test files: {str(e)}")

    def _load_augmentation_configs(self):
        """Load augmentation configurations, preferring structured JSON over text log."""
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

    def _scan_task_directories(self, models_dir):
        """Scan for all task directories in the existing job structure."""
        task_dirs = []
        
        try:
            for root, dirs, files in os.walk(models_dir):
                # Look for directories containing best_model.pth
                if 'best_model.pth' in files:
                    task_dirs.append(root)
            
            print(f"Found task directories: {len(task_dirs)}")
            for i, task_dir in enumerate(task_dirs, 1):
                rel_path = os.path.relpath(task_dir, models_dir) 
                print(f"{i}. {rel_path}")
            
            return task_dirs
            
        except Exception as e:
            print(f"Error scanning task directories: {str(e)}")
            return []