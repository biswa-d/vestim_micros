#!/usr/bin/env python3
"""
Test script to verify the denormalization process for training losses
"""
import os
import sys
import pandas as pd
import joblib
import numpy as np

# Paths
job_folder = r"c:\Users\dehuryb\code\vestim_micros\output\job_20250912-153351"
scaler_path = os.path.join(job_folder, "scalers", "augmentation_scaler.joblib")
model_path = os.path.join(job_folder, "models", "FNN_112_56", "B4096_LR_SLR_VP120_rep_1")
training_csv_path = os.path.join(model_path, "logs", "training_progress.csv")

print("=== Testing Denormalization Process ===")
print(f"Job folder: {job_folder}")
print(f"Scaler path: {scaler_path}")
print(f"Training CSV path: {training_csv_path}")
print()

# 1. Load and inspect the scaler
print("1. Loading scaler...")
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    print(f"   Scaler type: {type(scaler)}")
    print(f"   Number of features: {scaler.n_features_in_}")
    if hasattr(scaler, 'feature_names_in_'):
        print(f"   Feature names: {list(scaler.feature_names_in_)}")
        target_idx = list(scaler.feature_names_in_).index('voltage') if 'voltage' in scaler.feature_names_in_ else -1
        print(f"   Voltage column index: {target_idx}")
    else:
        print("   No feature names available")
        target_idx = -1  # Assume last column
    print(f"   Data min: {scaler.data_min_}")
    print(f"   Data max: {scaler.data_max_}")
    print()
else:
    print("   ERROR: Scaler file not found!")
    exit()

# 2. Load and inspect training progress CSV
print("2. Loading training progress CSV...")
if os.path.exists(training_csv_path):
    df = pd.read_csv(training_csv_path, comment='#')
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    # Get a specific loss value for testing
    test_epoch = 10
    if len(df) >= test_epoch:
        test_mse_norm = df.loc[test_epoch-1, 'val_loss_norm']  # 0-indexed
        print(f"   Test case - Epoch {test_epoch}: val_loss_norm = {test_mse_norm}")
    else:
        test_mse_norm = df['val_loss_norm'].min()
        print(f"   Test case - Best val_loss_norm = {test_mse_norm}")
    print()
else:
    print("   ERROR: Training CSV file not found!")
    exit()

# 3. Manual denormalization step-by-step
print("3. Manual denormalization process...")

# Step 3a: Convert MSE to RMSE
test_rmse_norm = np.sqrt(max(0, test_mse_norm))
print(f"   Step 3a: MSE to RMSE conversion")
print(f"            MSE (norm): {test_mse_norm}")
print(f"            RMSE (norm): {test_rmse_norm}")
print()

# Step 3b: Create dummy input for scaler
print(f"   Step 3b: Prepare scaler input")
dummy_input = np.zeros((1, scaler.n_features_in_))
print(f"            Dummy input shape: {dummy_input.shape}")

# Determine target column index
if hasattr(scaler, 'feature_names_in_') and 'voltage' in scaler.feature_names_in_:
    target_idx = list(scaler.feature_names_in_).index('voltage')
else:
    target_idx = scaler.n_features_in_ - 1  # Last column
print(f"            Target index: {target_idx}")

dummy_input[0, target_idx] = test_rmse_norm
print(f"            Dummy input[{target_idx}] = {test_rmse_norm}")
print()

# Step 3c: Apply inverse transform
print(f"   Step 3c: Apply scaler inverse transform")
denormalized = scaler.inverse_transform(dummy_input)
denormalized_rmse = denormalized[0, target_idx]
print(f"            Denormalized RMSE: {denormalized_rmse}")
print(f"            Denormalized RMSE (mV): {denormalized_rmse * 1000:.2f} mV")
print()

# 4. Compare with expected values
print("4. Comparison with expected behavior...")
print("   From your log example:")
print("   - Best Val Loss (Norm): 0.000193 (MSE)")
print("   - GUI Best Val RMSE: 13.8771 RMS Error [mV]")
print()

# Test with the specific value from the log
log_mse_norm = 0.000193
log_rmse_norm = np.sqrt(log_mse_norm)
dummy_log = np.zeros((1, scaler.n_features_in_))
dummy_log[0, target_idx] = log_rmse_norm
denorm_log = scaler.inverse_transform(dummy_log)
denorm_log_rmse = denorm_log[0, target_idx]

print(f"   Log example calculation:")
print(f"   - MSE norm: {log_mse_norm}")
print(f"   - RMSE norm: {log_rmse_norm:.6f}")
print(f"   - Denormalized RMSE: {denorm_log_rmse:.6f}")
print(f"   - Denormalized RMSE (mV): {denorm_log_rmse * 1000:.4f} mV")
print(f"   - Expected: ~13.8771 mV")
print(f"   - Match: {'YES' if abs(denorm_log_rmse * 1000 - 13.8771) < 1.0 else 'NO'}")
print()

# 5. Test our current implementation
print("5. Testing current implementation logic...")
sys.path.append('vestim/gui/src')
from standalone_testing_gui_qt import VEstimStandaloneTestingGUI

# Create a test GUI instance
gui = VEstimStandaloneTestingGUI(job_folder)
training_metrics = gui.get_training_metrics(os.path.join(model_path, "best_model.pth"), "voltage")

if training_metrics:
    print(f"   Implementation result:")
    print(f"   - Best Train Loss: {training_metrics.get('Best Train Loss', 'N/A')}")
    print(f"   - Best Val Loss: {training_metrics.get('Best Val Loss', 'N/A')}")
    print(f"   - Units: {training_metrics.get('Best Val Loss Unit', 'N/A')}")
else:
    print("   ERROR: Could not get training metrics from implementation")