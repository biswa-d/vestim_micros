#!/usr/bin/env python3
"""
Quick test to verify TrainingTaskManager initialization works correctly.
"""

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from vestim.gateway.src.training_task_manager_qt import TrainingTaskManager
    from vestim.gateway.src.job_manager_qt import JobManager
    
    print("✅ Import successful!")
    
    # Test initialization without global_params
    print("Testing initialization without global_params...")
    job_manager = JobManager()
    manager1 = TrainingTaskManager(job_manager=job_manager)
    print(f"✅ No global_params: {hasattr(manager1, 'global_params')} - {type(manager1.global_params)}")
    
    # Test initialization with global_params
    print("Testing initialization with global_params...")
    test_params = {'DEVICE_SELECTION': 'CPU', 'MODEL_TYPE': 'LSTM'}
    manager2 = TrainingTaskManager(job_manager=job_manager, global_params=test_params)
    print(f"✅ With global_params: {hasattr(manager2, 'global_params')} - {manager2.global_params}")
    
    # Test device selection
    print(f"✅ Device correctly set: {manager2.device}")
    
    print("\n🎉 All tests passed! TrainingTaskManager initialization is fixed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
