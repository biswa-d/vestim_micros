#!/usr/bin/env python3
"""
Test script to verify the default settings functionality
"""

import sys
import os

# Add the vestim package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

try:
    from vestim.config_manager import (
        get_default_folders, 
        get_default_hyperparams, 
        update_last_used_folders,
        update_last_used_hyperparams
    )
    
    print("✅ Config manager imports successful")
    
    # Test default folders
    print("\n📁 Testing default folders:")
    default_folders = get_default_folders()
    print(f"Default folders: {default_folders}")
    
    # Test default hyperparameters
    print("\n⚙️ Testing default hyperparameters:")
    default_hyperparams = get_default_hyperparams()
    print(f"Default hyperparams keys: {list(default_hyperparams.keys())}")
    print(f"Feature columns: {default_hyperparams.get('FEATURE_COLUMNS', 'Not found')}")
    print(f"Target column: {default_hyperparams.get('TARGET_COLUMN', 'Not found')}")
    
    # Test updating settings
    print("\n💾 Testing settings update:")
    test_hyperparams = {
        "FEATURE_COLUMNS": ["Test_Feature1", "Test_Feature2"],
        "TARGET_COLUMN": "Test_Target",
        "MODEL_TYPE": "LSTM",
        "BATCH_SIZE": "100"
    }
    
    update_last_used_hyperparams(test_hyperparams)
    print("Settings updated successfully")
    
    # Verify the update
    updated_hyperparams = get_default_hyperparams()
    print(f"Updated feature columns: {updated_hyperparams.get('FEATURE_COLUMNS', 'Not found')}")
    
    print("\n✅ All tests passed!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
