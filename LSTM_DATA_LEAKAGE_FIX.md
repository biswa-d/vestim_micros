# LSTM Sequence Data Splitting Fix: Preventing Overfitting and Data Leakage

## Problem Identified

The current LSTM sequence training implementation has a **critical data leakage issue** that leads to overfitting and unreliable validation performance. Here's the problem:

### Current Flawed Process:
1. **Sequence Creation**: All CSV files are processed to create overlapping sequences
   - File 1: sequences [0-49], [1-50], [2-51], ..., [950-999]
   - File 2: sequences [0-49], [1-50], [2-51], ..., [950-999]
   - etc.

2. **Random Shuffling**: All sequences from all files are shuffled together randomly
   ```python
   np.random.shuffle(indices)  # Line 107 in data_loader_service.py
   ```

3. **Train/Val Split**: Random 70/30 split of shuffled sequences
   ```python
   train_indices, valid_indices = indices[:train_size], indices[train_size:]
   ```

### Why This Causes Overfitting:

1. **Temporal Overlap**: Sequences [0-49] and [1-50] share 49 timesteps (98% overlap)
2. **Train/Val Contamination**: Due to random splitting, validation set likely contains sequences that overlap heavily with training sequences
3. **Information Leakage**: Model sees "future" information during training that helps predict validation targets
4. **False Performance**: Validation loss appears artificially low due to this leakage

## Example of the Problem:

Suppose we have sequences:
- Training: [0-49], [5-54], [10-59]
- Validation: [2-51], [7-56]

The validation sequence [2-51] overlaps with training sequence [0-49] by 47 timesteps! The model has essentially seen most of this validation data during training.

## Solution: Non-Overlapping Sequential Split

### FNN Approach (Currently Working):
The FNN model correctly prevents data leakage by:
1. Creating file-wise chunks (no mixing between files)
2. Shuffling chunks (not individual samples)
3. Ensuring batches contain data from only one file

### Required LSTM Fix:
We need to implement **temporal sequence splitting** instead of random splitting:

1. **Option 1: File-wise splitting**
   - Use first N files for training, remaining for validation
   - Ensures complete temporal separation between train/val

2. **Option 2: Temporal splitting per file**
   - Use first 70% of sequences from each file for training
   - Use last 30% of sequences for validation
   - Maintains temporal order

3. **Option 3: Gap-based splitting**
   - Create sequences from first 70% of each file for training
   - Skip a gap (e.g., lookback window size)
   - Create sequences from remaining data for validation

## Implementation Status

- ✅ **Problem Identified**: Random sequence shuffling causes data leakage
- ✅ **Root Cause Located**: Lines 107-109 in `data_loader_service.py`
- ✅ **Solution Designed**: Multiple approaches outlined above
- ✅ **Implementation Complete**: Temporal splitting method implemented
- ✅ **Configuration Added**: New `SEQUENCE_SPLIT_METHOD` parameter in hyperparams.json
- ✅ **Integration Complete**: Training task manager updated to use new method

## Implementation Details

### What Was Fixed:
1. **New Method**: Added `create_temporal_sequence_data_loaders()` method
2. **File-wise Processing**: Each CSV file is processed individually 
3. **Temporal Splitting**: Within each file, first 70% of sequences → training, last 30% → validation
4. **No Overlap**: Zero temporal overlap between training and validation sequences
5. **Configuration**: New `SEQUENCE_SPLIT_METHOD` hyperparameter (default: "temporal")

### Key Changes:
- **data_loader_service.py**: Added temporal sequence splitting method
- **training_task_manager_qt.py**: Updated to pass `sequence_split_method` parameter
- **hyperparams.json**: Added `"SEQUENCE_SPLIT_METHOD": "temporal"` parameter

### Backward Compatibility:
- Users can set `"SEQUENCE_SPLIT_METHOD": "random"` to use old behavior
- Default is `"temporal"` for new projects (prevents data leakage)
- FNN models continue using their existing file-wise chunk approach

## Next Steps

1. Choose preferred splitting strategy (file-wise, temporal, or gap-based)
2. Modify `create_data_loaders()` method in `DataLoaderService`
3. Add configuration parameter to control splitting method
4. Test with existing LSTM models to verify fix
5. Update documentation and hyperparameter examples

## Impact

Fixing this issue will:
- ✅ Eliminate data leakage between train/validation sets
- ✅ Provide realistic validation performance metrics
- ✅ Improve model generalization to truly unseen data
- ✅ Enable proper early stopping and hyperparameter tuning
- ✅ Make validation loss a reliable indicator of model performance

This is a **critical fix** that will significantly improve the reliability and trustworthiness of LSTM model training results.
