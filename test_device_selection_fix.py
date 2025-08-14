#!/usr/bin/env python3
"""
Test script to verify that the device selection parameter fix is working.
This test simulates the parameter flow from GUI to training task creation.
"""

import sys
import os

# Add the vestim module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from vestim.gateway.src.training_setup_manager_qt import TrainingSetupManager
from vestim.utils.logger_config import get_logger
import tempfile

def test_device_selection_parameter_flow():
    """Test that DEVICE_SELECTION parameter flows correctly from GUI params to final task"""
    
    logger = get_logger(__name__)
    print("=== Testing Device Selection Parameter Flow ===")
    
    # Create a temporary job directory
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Simulate GUI parameters with CUDA device selection
        gui_params = {
            'DEVICE_SELECTION': 'cuda:0',  # This is what GUI sends
            'BATCH_SIZE': 32,
            'MAX_EPOCHS': 100,
            'INITIAL_LR': 0.001,
            'VALID_PATIENCE': 10,
            'VALID_FREQUENCY': 1,
            'SCHEDULER_TYPE': 'StepLR',
            'REPETITIONS': 1,
            'LOOKBACK': 30,
            'NUM_WORKERS': 2
        }
        
        # Create a mock model task
        mock_model_task = {
            'model': None,  # We don't need actual model for this test
            'model_dir': temp_dir,
            'model_type': 'LSTM',
            'hyperparams': {
                'INPUT_SIZE': 10,
                'OUTPUT_SIZE': 1,
                'HIDDEN_UNITS': 64,
                'LAYERS': 2,
                'LOOKBACK': 30,
                'model_path': os.path.join(temp_dir, 'model.pth')
            },
            'FEATURE_COLUMNS': ['feature1', 'feature2'],
            'TARGET_COLUMN': 'target'
        }
        
        try:
            # Create the training setup manager
            setup_manager = TrainingSetupManager(params=gui_params, job_directory=temp_dir)
            
            # Call the _create_task_info method directly to test parameter flow
            task_info = setup_manager._create_task_info(
                model_task=mock_model_task,
                hyperparams=gui_params,  # This should contain DEVICE_SELECTION: 'cuda:0'
                repetition=1
            )
            
            # Check if DEVICE_SELECTION made it into the final hyperparams
            final_device = task_info['hyperparams'].get('DEVICE_SELECTION')
            final_num_workers = task_info['data_loader_params'].get('num_workers')
            
            print(f"✓ Original GUI params DEVICE_SELECTION: {gui_params.get('DEVICE_SELECTION')}")
            print(f"✓ Final task hyperparams DEVICE_SELECTION: {final_device}")
            print(f"✓ Original GUI params NUM_WORKERS: {gui_params.get('NUM_WORKERS')}")
            print(f"✓ Final task data_loader_params num_workers: {final_num_workers}")
            
            # Verify the fix worked
            if final_device == 'cuda:0':
                print("✅ SUCCESS: DEVICE_SELECTION parameter flows correctly from GUI to training task!")
            else:
                print(f"❌ FAILED: Expected DEVICE_SELECTION='cuda:0', got '{final_device}'")
                return False
                
            if final_num_workers == 2:
                print("✅ SUCCESS: NUM_WORKERS parameter flows correctly from GUI to training task!")
            else:
                print(f"❌ FAILED: Expected NUM_WORKERS=2, got {final_num_workers}")
                return False
                
            return True
            
        except Exception as e:
            print(f"❌ ERROR during testing: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = test_device_selection_parameter_flow()
    if success:
        print("\n🎉 All tests passed! The device selection fix is working correctly.")
    else:
        print("\n💥 Tests failed. The fix needs more work.")
    
    sys.exit(0 if success else 1)
