#!/usr/bin/env python3
"""
Test script for updated standalone testing GUI functionality
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vestim.gui.src.standalone_testing_gui_qt import VEstimStandaloneTestingGUI
from PyQt5.QtWidgets import QApplication

def test_training_metrics():
    """Test the training metrics extraction"""
    
    # Test with a sample model path
    model_path = "output/job_20250904-123450/models/FNN_64_32/B4096_LR_SLR_VP50_rep_1/FNN_64_32_B4096_LR_SLR_VP50_rep_1_model.h5"
    
    app = QApplication([])
    gui = VEstimStandaloneTestingGUI("output/job_20250904-123450")
    
    # Test training metrics extraction
    print("Testing training metrics extraction...")
    training_metrics = gui.get_training_metrics(model_path)
    
    if training_metrics:
        print("✓ Training metrics extracted successfully:")
        for key, value in training_metrics.items():
            print(f"  {key}: {value}")
    else:
        print("❌ No training metrics found")
    
    # Test with dummy result data
    print("\nTesting result display...")
    dummy_result = {
        'model_type': 'FNN',
        'architecture': 'FNN_64_32',
        'task': 'B4096_LR_SLR_VP50_rep_1',
        'target_column': 'voltage',
        'model_file_path': model_path,
        'MAE': 0.0025,
        'MSE': 0.0000075,
        'RMSE': 0.0027,
        'MAPE': 0.15,
        'R²': 0.98,
        'predictions_file': 'output/job_20250904-123450/test_results/predictions.csv'
    }
    
    gui.add_result_row(dummy_result)
    print("✓ Result row added successfully")
    
    app.quit()

if __name__ == "__main__":
    test_training_metrics()