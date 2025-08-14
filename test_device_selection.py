#!/usr/bin/env python3
"""
Test script to verify device selection works correctly from hyperparameters to training.
"""

import torch
import sys
import os

# Add the project root to the Python path so we can import vestim modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vestim.gateway.src.training_task_manager_qt import TrainingTaskManager
from vestim.gateway.src.job_manager_qt import JobManager

def test_device_selection():
    """Test that device selection from global_params correctly flows to training services."""
    
    print("🔍 Testing Device Selection from Hyperparameters")
    print("=" * 60)
    
    # Test 1: CPU selection
    print("\n1. Testing CPU selection:")
    cpu_params = {'DEVICE_SELECTION': 'CPU'}
    job_manager = JobManager()
    training_manager = TrainingTaskManager(job_manager=job_manager, global_params=cpu_params)
    
    print(f"   - Global params device: {cpu_params['DEVICE_SELECTION']}")
    print(f"   - TrainingTaskManager device: {training_manager.device}")
    print(f"   - Training service device: {training_manager.training_service.device}")
    
    assert str(training_manager.device) == 'cpu', f"Expected CPU device, got {training_manager.device}"
    assert str(training_manager.training_service.device) == 'cpu', f"Expected CPU in service, got {training_manager.training_service.device}"
    print("   ✅ CPU selection works correctly!")
    
    # Test 2: CUDA selection (if available)
    if torch.cuda.is_available():
        print("\n2. Testing CUDA selection:")
        cuda_params = {'DEVICE_SELECTION': 'cuda:0'}
        training_manager_cuda = TrainingTaskManager(job_manager=job_manager, global_params=cuda_params)
        
        print(f"   - Global params device: {cuda_params['DEVICE_SELECTION']}")
        print(f"   - TrainingTaskManager device: {training_manager_cuda.device}")
        print(f"   - Training service device: {training_manager_cuda.training_service.device}")
        
        assert 'cuda' in str(training_manager_cuda.device), f"Expected CUDA device, got {training_manager_cuda.device}"
        assert 'cuda' in str(training_manager_cuda.training_service.device), f"Expected CUDA in service, got {training_manager_cuda.training_service.device}"
        print("   ✅ CUDA selection works correctly!")
        
        # Test 3: CUDA Graphs with CUDA device
        print("\n3. Testing CUDA Graphs with CUDA device:")
        cuda_graphs_params = {
            'DEVICE_SELECTION': 'cuda:0',
            'USE_CUDA_GRAPHS': True,
            'MODEL_TYPE': 'FNN'
        }
        training_manager_graphs = TrainingTaskManager(job_manager=job_manager, global_params=cuda_graphs_params)
        
        print(f"   - Global params device: {cuda_graphs_params['DEVICE_SELECTION']}")
        print(f"   - TrainingTaskManager device: {training_manager_graphs.device}")
        print(f"   - Training service type: {type(training_manager_graphs.training_service).__name__}")
        print(f"   - Training service device: {training_manager_graphs.training_service.device}")
        
        assert 'cuda' in str(training_manager_graphs.device), f"Expected CUDA device, got {training_manager_graphs.device}"
        assert 'cuda' in str(training_manager_graphs.training_service.device), f"Expected CUDA in graphs service, got {training_manager_graphs.training_service.device}"
        print("   ✅ CUDA Graphs selection works correctly!")
        
    else:
        print("\n2. CUDA not available - skipping CUDA tests")
    
    # Test 4: Invalid device should fallback to CPU
    print("\n4. Testing invalid device fallback:")
    invalid_params = {'DEVICE_SELECTION': 'invalid_device'}
    training_manager_invalid = TrainingTaskManager(job_manager=job_manager, global_params=invalid_params)
    
    print(f"   - Global params device: {invalid_params['DEVICE_SELECTION']}")
    print(f"   - TrainingTaskManager device: {training_manager_invalid.device}")
    print(f"   - Training service device: {training_manager_invalid.training_service.device}")
    
    # Should fallback to available device
    expected_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"   - Expected fallback device: {expected_device}")
    assert str(training_manager_invalid.device) == expected_device, f"Expected {expected_device} fallback, got {training_manager_invalid.device}"
    print("   ✅ Invalid device fallback works correctly!")
    
    print("\n" + "=" * 60)
    print("🎉 All device selection tests passed!")
    print("\n✅ Device selection from hyperparameters is working correctly!")
    print("✅ Training services properly respect the selected device!")
    print("✅ Your CPU selection issue should now be fixed!")

if __name__ == "__main__":
    test_device_selection()
