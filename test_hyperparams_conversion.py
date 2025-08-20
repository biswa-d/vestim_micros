#!/usr/bin/env python3
"""
Test script to validate model-type aware hyperparameter conversion.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vestim.gateway.src.training_task_manager_qt import TrainingTaskManager

def test_lstm_conversion():
    """Test LSTM hyperparameter conversion."""
    print("Testing LSTM hyperparameter conversion...")
    
    lstm_hyperparams = {
        'MODEL_TYPE': 'LSTM',
        'LAYERS': '2',
        'HIDDEN_UNITS': '64',
        'BATCH_SIZE': '32',
        'MAX_EPOCHS': '100',
        'INITIAL_LR': '0.001',
        'SCHEDULER_TYPE': 'StepLR',
        'LR_PERIOD': '50',
        'LR_PARAM': '0.5',
        'VALID_PATIENCE': '10',
        'ValidFrequency': '5',
        'LOOKBACK': '50',
        'REPETITIONS': '1'
    }
    
    manager = TrainingTaskManager()
    try:
        converted = manager.convert_hyperparams(lstm_hyperparams.copy())
        
        # Check that LSTM-specific params were converted
        assert isinstance(converted['LAYERS'], int), f"LAYERS should be int, got {type(converted['LAYERS'])}"
        assert isinstance(converted['HIDDEN_UNITS'], int), f"HIDDEN_UNITS should be int, got {type(converted['HIDDEN_UNITS'])}"
        assert converted['LAYERS'] == 2, f"LAYERS should be 2, got {converted['LAYERS']}"
        assert converted['HIDDEN_UNITS'] == 64, f"HIDDEN_UNITS should be 64, got {converted['HIDDEN_UNITS']}"
        
        # Check common params
        assert isinstance(converted['BATCH_SIZE'], int), f"BATCH_SIZE should be int, got {type(converted['BATCH_SIZE'])}"
        assert isinstance(converted['MAX_EPOCHS'], int), f"MAX_EPOCHS should be int, got {type(converted['MAX_EPOCHS'])}"
        assert isinstance(converted['INITIAL_LR'], float), f"INITIAL_LR should be float, got {type(converted['INITIAL_LR'])}"
        
        print("✓ LSTM hyperparameter conversion passed!")
        return True
        
    except Exception as e:
        print(f"✗ LSTM hyperparameter conversion failed: {e}")
        return False

def test_fnn_conversion():
    """Test FNN hyperparameter conversion."""
    print("Testing FNN hyperparameter conversion...")
    
    fnn_hyperparams = {
        'MODEL_TYPE': 'FNN',
        'HIDDEN_LAYER_SIZES': '128,64,32',  # FNN uses this instead of LAYERS/HIDDEN_UNITS
        'DROPOUT_PROB': '0.2',
        'BATCH_SIZE': '64',
        'MAX_EPOCHS': '200',
        'INITIAL_LR': '0.0001',
        'SCHEDULER_TYPE': 'StepLR',
        'LR_PERIOD': '100',
        'LR_PARAM': '0.8',
        'VALID_PATIENCE': '15',
        'ValidFrequency': '10',
        'LOOKBACK': '100',
        'REPETITIONS': '3'
    }
    
    manager = TrainingTaskManager()
    try:
        converted = manager.convert_hyperparams(fnn_hyperparams.copy())
        
        # Check that FNN-specific params were NOT converted (should remain as strings)
        assert isinstance(converted['HIDDEN_LAYER_SIZES'], str), f"HIDDEN_LAYER_SIZES should remain str, got {type(converted['HIDDEN_LAYER_SIZES'])}"
        assert converted['HIDDEN_LAYER_SIZES'] == '128,64,32', f"HIDDEN_LAYER_SIZES should be '128,64,32', got {converted['HIDDEN_LAYER_SIZES']}"
        
        # Check that LSTM-specific params don't exist
        assert 'LAYERS' not in converted or converted.get('LAYERS') is None, "LAYERS should not be in FNN hyperparams"
        assert 'HIDDEN_UNITS' not in converted or converted.get('HIDDEN_UNITS') is None, "HIDDEN_UNITS should not be in FNN hyperparams"
        
        # Check common params
        assert isinstance(converted['BATCH_SIZE'], int), f"BATCH_SIZE should be int, got {type(converted['BATCH_SIZE'])}"
        assert isinstance(converted['MAX_EPOCHS'], int), f"MAX_EPOCHS should be int, got {type(converted['MAX_EPOCHS'])}"
        assert isinstance(converted['INITIAL_LR'], float), f"INITIAL_LR should be float, got {type(converted['INITIAL_LR'])}"
        
        print("✓ FNN hyperparameter conversion passed!")
        return True
        
    except Exception as e:
        print(f"✗ FNN hyperparameter conversion failed: {e}")
        return False

def test_gru_conversion():
    """Test GRU hyperparameter conversion."""
    print("Testing GRU hyperparameter conversion...")
    
    gru_hyperparams = {
        'MODEL_TYPE': 'GRU',
        'LAYERS': '3',
        'HIDDEN_UNITS': '128',
        'BATCH_SIZE': '16',
        'MAX_EPOCHS': '150',
        'INITIAL_LR': '0.002',
        'SCHEDULER_TYPE': 'ReduceLROnPlateau',
        'PLATEAU_PATIENCE': '5',
        'PLATEAU_FACTOR': '0.3',
        'VALID_PATIENCE': '20',
        'ValidFrequency': '3',
        'LOOKBACK': '75',
        'REPETITIONS': '2'
    }
    
    manager = TrainingTaskManager()
    try:
        converted = manager.convert_hyperparams(gru_hyperparams.copy())
        
        # Check that GRU-specific params were converted (same as LSTM)
        assert isinstance(converted['LAYERS'], int), f"LAYERS should be int, got {type(converted['LAYERS'])}"
        assert isinstance(converted['HIDDEN_UNITS'], int), f"HIDDEN_UNITS should be int, got {type(converted['HIDDEN_UNITS'])}"
        assert converted['LAYERS'] == 3, f"LAYERS should be 3, got {converted['LAYERS']}"
        assert converted['HIDDEN_UNITS'] == 128, f"HIDDEN_UNITS should be 128, got {converted['HIDDEN_UNITS']}"
        
        # Check scheduler params for ReduceLROnPlateau
        assert isinstance(converted['PLATEAU_PATIENCE'], int), f"PLATEAU_PATIENCE should be int, got {type(converted['PLATEAU_PATIENCE'])}"
        assert isinstance(converted['PLATEAU_FACTOR'], float), f"PLATEAU_FACTOR should be float, got {type(converted['PLATEAU_FACTOR'])}"
        
        print("✓ GRU hyperparameter conversion passed!")
        return True
        
    except Exception as e:
        print(f"✗ GRU hyperparameter conversion failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Model-Type Aware Hyperparameter Conversion Test")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 3
    
    # Run tests
    if test_lstm_conversion():
        tests_passed += 1
    if test_fnn_conversion():
        tests_passed += 1
    if test_gru_conversion():
        tests_passed += 1
    
    print("=" * 60)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Hyperparameter conversion is working correctly.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please check the implementation.")
        sys.exit(1)
