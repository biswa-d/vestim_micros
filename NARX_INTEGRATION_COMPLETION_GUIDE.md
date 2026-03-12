# NARX Model Integration - Completion Guide

**Status**: Foundation complete on `narx_filter_pad` branch. Core components in place.

## ✅ COMPLETED

### Phase 1: Core Model Implementation
- ✅ Created `NARX_model.py` with:
  - Single-step forward pass: `forward(x_current, y_previous=None)`
  - Sequence mode: `forward_sequence(x_sequence, y_init=None)`
  - Xavier weight initialization
  - Support for dropout, activation functions, layer normalization

### Phase 2: Filter-Based Padding
- ✅ Created `filter_padding_calculator.py` with:
  - `FilterPaddingCalculator.calculate_padding_length()` - computes padding from filter time constants
  - `FilterPaddingCalculator.calculate_time_constant()` - supports Butterworth, Moving Average, EMA, Savitzky-Golay
  - Empirically-tuned filter coefficients (5x for Butterworth, 8x for EMA, etc.)

### Phase 3: Data Augmentation Integration
- ✅ Added to `data_augment_service.py`:
  - `remove_padding()` - remove padded rows after filtering
  - `apply_filter_with_padding()` - complete workflow: pad → filter → remove-pad
  - Returns metadata about padding applied for later removal in dataloaders

### Phase 4: Hyperparameter GUI
- ✅ Updated `hyper_param_gui_qt.py`:
  - Added "NARX" to model_combo dropdown
  - Added NARX-specific parameter fields:
    - HIDDEN_LAYER_SIZES
    - DROPOUT_PROB
    - OUTPUT_DELAY (autoregressive order)
    - activation function selector
- ✅ Created `defaults_templates/hyperparams_narx.json` template

### Phase 5: Training Integration (Partial)
- ✅ Updated `training_task_service.py`:
  - Added NARX cases to `train_epoch()` method
  - Added NARX cases to `validate_epoch()` method
  - NARX handled like FNN (no hidden states, stateless training)
  
### Phase 6: Testing Integration (Partial)
- ✅ Updated `standalone_testing_manager_qt.py`:
  - Added NARXModel import
  - Added NARX model instantiation in `_test_task_model()`
  - Can now load and run NARX models during standalone testing

---

## 🟡 REMAINING WORK (Priority Order)

### HIGH PRIORITY - Required for end-to-end training

#### 1. **Model Instantiation in Training (5-10 min)**
**File**: `vestim/gateway/src/standalone_testing_manager_qt.py` (DONE)
**Files Still Needed**:
- `vestim/services/model_testing/src/testing_service.py` - Add NARX case in model creation
- `vestim/services/model_testing/src/continuous_testing_service.py` - Add NARX case  
- `vestim/services/model_training/src/FNN_model_service.py` - Add NARX initialization wrapper (if used)

**Action**: Find model creation sections (look for `if model_type == 'FNN'` patterns) and add:
```python
elif model_type == 'NARX':
    model = NARXModel(
        input_size=input_size,
        output_size=output_size,
        hidden_layer_sizes=hidden_sizes,
        output_delay=output_delay,
        dropout_prob=dropout_prob,
        activation_function=activation_fn,
        device=device
    )
```

####  2. **Data Loader Integration (30-45 min)**
**Files**: 
- `vestim/services/model_training/src/data_loader_service.py`

**Action**: 
1. Add NARX to training method options (check for `if training_method == 'Sequential'` patterns)
2. For NARX, treat like FNN (stateless, no sequence unwrapping)
3. Handle `OUTPUT_DELAY` parameter - only matters if we stack y_prev into feature matrix
   - **For now**: Treat NARX sequences like FNN: batch size is 1, lookback=1
   - **Future enhancement**: Implement sliding window for y_previous construction

3. Update padding removal logic:
   - When augmentation metadata contains `padding_info`, remove that many rows from beginning of train/val batches
   - Add this in `collate_fn` or after data loading

#### 3. **Update adaptive_gui_utils.py (5-10 min)**
**File**: `vestim/gui/src/adaptive_gui_utils.py`

**Action**: Add NARX to training method checks (currently only handles LSTM/GRU separately from FNN)
```python
if model_type in ['LSTM', 'GRU', 'LSTM_EMA', 'LSTM_LPF']:
    # RNN logic
elif model_type ==  'FNN' or model_type == 'NARX':
    # Stateless logic
```

#### 4. **Update Training Method GUI (5 min)**
**File**: `vestim/gui/src/hyper_param_gui_qt.py`

**Action**: In `update_training_method()`, add NARX validation:
```python
if selected_model == 'NARX':
    # Show only 'Sequential' option (NARX doesn't support WholeSequence)
    self.training_method_combo.setEnabled(False)
    self.training_method_combo.clear()
    self.training_method_combo.addItem('Sequential')
```

### MEDIUM PRIORITY - Feature Completeness

#### 5. **Padding Removal During Training (20-30 min)**
**Files**:
- `vestim/services/model_training/src/data_loader_service.py` (collate_fn)
- `vestim/services/model_training/src/sequence_rnn_data_handler.py` or FNN handler

**Action**:
- Read `augmentation_metadata.json` from job folder
- Extract `padding_info.length` 
- Pass to data loader
- In batch creation, skip first N rows if padding was applied

**Pseudocode**:
```python
if augmentation_metadata.get('padding', {}).get('applied'):
    padding_length = augmentation_metadata['padding']['length']
    # Skip first padding_length rows in training data
    train_df = train_df.iloc[padding_length:]
    val_df = val_df.iloc[padding_length:]
```

#### 6. **Update Testing Data Loader (20-30 min)**
**Files**:
- `vestim/gateway/src/standalone_testing_manager_qt.py` - Already calculates padding, needs to remove it before inference
- Or create a test data handler that mirrors training handler

**Action**: In stanalone_testing_manager_qt.py, after augmentation but before model runs:
```python
# Remove padding from test data before inference
if padding_applied_during_augmentation:
    test_df = test_df.iloc[padding_length:]
```

#### 7. **Integrate Filter-Based Padding into Augmentation Workflow (15-20 min)**
**File**: `vestim/services/data_processor/src/data_augment_service.py` augmentation methods

**Action**:
- When filters are applied, automatically calculate and apply padding first
- Call `apply_filter_with_padding()` instead of just `apply_butterworth_filter()`
- Store padding length in augmentation_metadata
- Option: Make automatic padding configurable (add UI toggle)

**Example integration point**:
```python
# In augmentation GUI or upload service
if filters_config:
    padding_length = FilterPaddingCalculator.calculate_padding_length(filters_config)
    df = data_augment_service.apply_filter_with_padding(
        df, 
        column_name,
        corner_frequency,
        sampling_rate,
        auto_padding=True
    )
```

### LOW PRIORITY - Polish & Validation

#### 8. **Update Config Manager (10 min)**
**File**: `vestim/config_manager.py`

**Action**: Add NARX to model type validation lists

#### 9. **Add Unit Tests (30-45 min)**
**Create**: `vestim/tests/test_narx_model.py`

```python
def test_narx_forward():
    model = NARXModel(input_size=5, output_size=1, hidden_layer_sizes=[64, 32], output_delay=1)
    x = torch.randn(32, 5)  # batch of 32, 5 features
    y = model(x)
    assert y.shape == (32, 1)

def test_narx_sequence():
    model = NARXModel(input_size=5, output_size=1, hidden_layer_sizes=[64, 32], output_delay=1)
    x_seq = torch.randn(32, 100, 5)  # batch of 32, sequence of 100, 5 features
    y_seq = model.forward_sequence(x_seq)
    assert y_seq.shape == (32, 100, 1)

def test_padding_calculator():
    filters = [{'type': 'butterworth', 'corner_frequency': 0.02, 'sampling_rate': 1.0}]
    pad_len = FilterPaddingCalculator.calculate_padding_length(filters)
    assert pad_len > 0
```

#### 10. **Add Documentation (15-20 min)**
**Create**: `NARX_MODEL_DOCUMENTATION.md`

- Model architecture overview
- Parameter tuning guide
- When to use NARX vs LSTM vs FNN
- Performance characteristics
- Example training scripts

---

## 🎯 Testing Checklist

Once all components integrated, test this workflow:

- [ ] GUI: Create hyperparameters with MODEL_TYPE="NARX"
- [ ] GUI: Set HIDDEN_LAYER_SIZES, OUTPUT_DELAY, activation
- [ ] GUI: Apply filters to test data
- [ ] Augmentation: Verify padding calculated and applied
- [ ] Training: Training starts without errors
- [ ] Training: Loss decreases over epochs
- [ ] Testing: Model inference completes
- [ ] Testing: Predictions are reasonable (not NaN/Inf)
- [ ] Validation: Filtered columns don't start from zero

---

## 📋 Implementation Notes

### OUTPUT_DELAY (Autoregressive Order)
- Currently NARX_model.forward() ignores y_previous and initializes to zeros
- **For MVP**: This is fine - model learns from pure exogenous inputs
- **Future Enhancement**: Modify data loader to construct y_previous from previous predictions during training (requires sequential processing or teacher forcing)

### Padding Removal Timing
- **Option A (Simpler, Current Plan)**: Remove padding after augmentation, before training/testing
- **Option B (Alternative)**: Keep padding until after data augmentation, then remove before dataloader creation
- Both work; Option A is cleaner for inference on external test data

### Training Method for NARX
- Should be "Sequential" only (don't use "WholeSequence"FNN mode)
- NARX per-timestep prediction naturally fits Sequential mode
- GUI should disable other options when NARX selected

---

## 📌 Key Files & Entry Points

| Task | File | Search String |
|------|------|---|
| Import NARX | Any service | `from NARX_model import NARXModel` |
| Model Creation | testing_service.py | `if model_type == 'FNN':` |
| Data Handler | data_loader_service.py | `create_fnn_batch_data_loaders` |
| Padding Removal | data_loader_service.py | `collate_fn` or batch create methods |
| Training Loop | training_task_service.py | `elif model_type == "FNN":` |  
| GUI Update | adaptive_gui_utils.py | `model_type = params.get('MODEL_TYPE'` |
| Config Validation | config_manager.py | `VALID_MODEL_TYPES` or similar |

---

## 🚀 Quick Start for Next Developer

1. Search for all `if model_type == 'FNN'` patterns in the codebase  
2. Add `elif model_type == 'NARX'` branches in each location
3. For NARX, mostly replicate FNN logic (stateless, no hidden states)
4. Update data loaders to handle padding removal via augmentation metadata
5. Test filter → padding → remove-padding → train workflow end-to-end

**Estimated Total Time**: 2-4 hours for complete integration + testing

