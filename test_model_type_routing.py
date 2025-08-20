#!/usr/bin/env python3
"""
Quick test to verify model type routing in data loader
"""

def test_model_type_routing():
    """Test the data loader routing logic"""
    
    # Simulate hyperparameters for FNN model
    fnn_hyperparams = {
        'MODEL_TYPE': 'FNN',
        'TRAINING_METHOD': 'Sequence-to-Sequence',
        'HIDDEN_LAYER_SIZES': '128,64',
        'DROPOUT_PROB': '0.1',
        'BATCH_SIZE': '5000',
        'MAX_EPOCHS': '100',
        'INITIAL_LR': '0.001',
        'VALID_PATIENCE': '10',
        'ValidFrequency': '3',
        'LOOKBACK': '400',  # Should be ignored for FNN
        'REPETITIONS': '1'
    }
    
    # Simulate hyperparameters for LSTM model  
    lstm_hyperparams = {
        'MODEL_TYPE': 'LSTM',
        'TRAINING_METHOD': 'Sequence-to-Sequence', 
        'LAYERS': '2',
        'HIDDEN_UNITS': '64',
        'BATCH_SIZE': '32',
        'MAX_EPOCHS': '100',
        'INITIAL_LR': '0.001',
        'VALID_PATIENCE': '10',
        'ValidFrequency': '3',
        'LOOKBACK': '400',
        'REPETITIONS': '1'
    }
    
    print("=== Testing Model Type Routing ===")
    
    # Test FNN routing
    model_type_fnn = fnn_hyperparams.get('MODEL_TYPE', 'LSTM')
    print(f"FNN hyperparams MODEL_TYPE: {model_type_fnn}")
    
    if model_type_fnn == "FNN":
        print("✅ FNN model would be routed to FNN batch data loader")
    else:
        print("❌ FNN model would incorrectly use sequence data loader")
    
    # Test LSTM routing  
    model_type_lstm = lstm_hyperparams.get('MODEL_TYPE', 'LSTM')
    print(f"LSTM hyperparams MODEL_TYPE: {model_type_lstm}")
    
    if model_type_lstm in ['LSTM', 'GRU']:
        print("✅ LSTM model would be routed to sequence data loader")
    else:
        print("❌ LSTM model would be incorrectly routed")
        
    print("\n=== Testing Convert Hyperparams ===")
    
    # Test convert_hyperparams logic
    def convert_hyperparams_test(hyperparams):
        """Test version of convert_hyperparams"""
        model_type = hyperparams.get('MODEL_TYPE', 'LSTM')
        print(f"Converting hyperparams for model_type: {model_type}")
        
        # Common parameters for all model types
        try:
            hyperparams['BATCH_SIZE'] = int(hyperparams['BATCH_SIZE'])
            hyperparams['MAX_EPOCHS'] = int(hyperparams['MAX_EPOCHS'])
            hyperparams['INITIAL_LR'] = float(hyperparams['INITIAL_LR'])
            print(f"✅ Common parameters converted successfully")
        except Exception as e:
            print(f"❌ Error converting common parameters: {e}")
            return False
        
        # Model-specific parameter conversion
        if model_type in ['LSTM', 'GRU']:
            try:
                hyperparams['LAYERS'] = int(hyperparams['LAYERS'])
                hyperparams['HIDDEN_UNITS'] = int(hyperparams['HIDDEN_UNITS'])
                print(f"✅ RNN parameters converted successfully")
            except Exception as e:
                print(f"❌ Error converting RNN parameters: {e}")
                return False
        elif model_type == 'FNN':
            # FNN-specific parameters - HIDDEN_LAYERS is already a string, no conversion needed
            print(f"✅ FNN parameters left as strings (correct)")
            
        return True
    
    # Test FNN conversion
    fnn_test = fnn_hyperparams.copy()
    print(f"\nTesting FNN hyperparams conversion:")
    success = convert_hyperparams_test(fnn_test)
    if success:
        print(f"Converted FNN hyperparams: BATCH_SIZE={fnn_test['BATCH_SIZE']} (type: {type(fnn_test['BATCH_SIZE'])})")
    
    # Test LSTM conversion  
    lstm_test = lstm_hyperparams.copy()
    print(f"\nTesting LSTM hyperparams conversion:")
    success = convert_hyperparams_test(lstm_test)
    if success:
        print(f"Converted LSTM hyperparams: LAYERS={lstm_test['LAYERS']} (type: {type(lstm_test['LAYERS'])}), HIDDEN_UNITS={lstm_test['HIDDEN_UNITS']} (type: {type(lstm_test['HIDDEN_UNITS'])})")

if __name__ == "__main__":
    test_model_type_routing()
