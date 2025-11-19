# VEstim - Voltage Estimation Modeling Tool

## Welcome to VEstim!

VEstim is a comprehensive machine learning tool for battery voltage estimation using LSTM, GRU, and FNN (Feed-Forward Neural Network) models. This tool provides an intuitive graphical interface for the complete machine learning workflow: data import, preprocessing, hyperparameter tuning, model training, and testing.

##   Quick Start Guide

### Your First Model Training

1. **Launch VEstim** - Double-click the VEstim executable
2. **Check Demo Data** - VEstim comes with sample battery data files ready to use
3. **Follow the 4-Step Workflow** below to train your first model in minutes!

---

## 📁 Understanding Your VEstim Project Folder

This folder contains:
- **`data/`** - Your training, validation, and test datasets
  - `train_data/` - Files for training models
  - `val_data/` - Files for validation during training
  - `test_data/` - Files for final model testing
- **`job_xxxxx/`** - Output folders created for each training session
- **`default_settings.json`** - Your saved preferences and settings
- **`README.md`** - This file!

### Demo Data Files

VEstim includes sample battery data files to help you get started:
- **`demo_train_data.csv`** - Sample training data (~762K data points)
- **`demo_validation_data.csv`** - Sample validation data
- **`demo_test_data.csv`** - Sample test data
- **`demo_combined_test_data.csv`** - Additional test data option

**Data Format**: All CSV files contain columns: `SOC`, `Current`, `Temp`, `Voltage`
- **SOC** - State of Charge (0-1)
- **Current** - Battery current (A)
- **Temp** - Temperature (°C)
- **Voltage** - Battery voltage (V) - *this is what we predict*

---

## 🔧 Complete Workflow Guide

### Step 1: Data Import & Organization

**GUI: Data Import Window**

1. **Launch VEstim** - The Data Import window opens first
2. **Review Pre-loaded Demo Data**:
   - Training files are automatically loaded from `data/train_data/`
   - Validation files are loaded from `data/val_data/`
   - Test files are loaded from `data/test_data/`
3. **File Format**: Select CSV (default) or MAT/XLSX if needed
4. **Add Your Own Data** (Optional):
   - Click "Select Train Folder" to choose your training data
   - Click "Select Validation Folder" for validation data
   - Click "Select Test Folder" for test data
   - Select files you want to use (Ctrl+click for multiple)

**What This Step Does**: Organizes your data files and prepares them for preprocessing.

**💡 Tip**: Start with the demo data to learn the tool, then replace with your own battery data later.

---

### Step 2: Data Preprocessing & Augmentation

**GUI: Data Augmentation Window**

1. **Configure Preprocessing Options**:
   -  **Enable data normalization** (Recommended - checked by default)
   - **Resampling**: Choose time interval for data resampling (e.g., 1 second)
   - **File Format**: Confirm output format (CSV recommended)

2. **Advanced Options**:
   - **Outlier Removal**: Remove data points beyond specified thresholds
   - **Data Validation**: Check for missing values and inconsistencies

3. **Process Data**: Click "Process Data" to clean and prepare your datasets

**What This Step Does**: 
- Normalizes data values to improve model training
- Resamples data to consistent time intervals
- Removes outliers and handles missing data
- Creates processed datasets ready for machine learning

**💡 Tip**: Data normalization significantly improves model performance - keep it enabled!

---

### Step 3: Hyperparameter Selection & Model Configuration

**GUI: Hyperparameter Selection Window**

#### Model Configuration
1. **Select Model Type**:
   - **LSTM** (Recommended) - Best for time-series data
   - **GRU** - Memory-efficient alternative to LSTM
   - **FNN** - Feed-forward network for simpler patterns

2. **Model Architecture**:
   - **Hidden Units**: Number of neurons (10-100, start with 10)
   - **Layers**: Number of hidden layers (1-3, start with 1)

#### Feature Selection
3. **Choose Input Features**:
   - Select from: `SOC`, `Current`, `Temp`
   - **Recommended**: Use all three for best performance
   - **Target**: `Voltage` (automatically selected)

#### Training Configuration
4. **Training Method**:
   - **Sequence-to-Sequence** (Default for LSTM/GRU)
   - **Whole Sequence** (Automatically converted to appropriate method)

5. **Training Parameters**:
   - **Lookback Window**: 400 (how many past points to consider)
   - **Batch Size**: 200 (number of samples per training batch)
   - **Max Epochs**: 5 (training iterations - start small)
   - **Learning Rate**: 0.0001 (how fast the model learns)

6. **Validation Settings**:
   - **Train/Validation Split**: 0.8 (80% train, 20% validation)
   - **Patience**: 10 (stop if no improvement for 10 epochs)

**What This Step Does**: Configures all model settings and training parameters.

**💡 Tip**: The default settings work well for most battery data. Adjust only if needed!

---

### Step 4: Model Training

**GUI: Training Setup Window**

1. **Review Configuration**: Verify all settings are correct
2. **Start Training**: Click "Start Training"
3. **Monitor Progress**: 
   - Watch training progress in real-time
   - View loss curves and validation metrics
   - Training stops automatically when optimal performance is reached

**Training Process**:
- Model learns patterns from your training data
- Validates performance on validation data
- Saves the best performing model automatically
- Creates detailed logs and metrics

**Output Location**: Results saved in `job_YYYYMMDD-HHMMSS/` folder

**💡 Tip**: Training time depends on data size and model complexity. Demo data typically trains in 1-5 minutes.

---

### Step 5: Model Testing & Evaluation

**GUI: Testing Window**

1. **Load Trained Model**: Select model from recent training job
2. **Select Test Data**: Choose test dataset (demo_test_data.csv)
3. **Run Predictions**: Click "Test Model" to generate predictions
4. **View Results**:
   - **Time Series Plot**: Actual vs. Predicted voltage over time
   - **Error Distribution**: Histogram of prediction errors
   - **Performance Metrics**: MAE, RMSE, R² score
   - **Improved Plotting**: Time axis in seconds, clear axis labels

**Evaluation Metrics**:
- **MAE** (Mean Absolute Error): Average prediction error
- **RMSE** (Root Mean Square Error): Penalizes large errors more
- **R²** (R-squared): How well model explains variance (closer to 1 = better)

**💡 Tip**: Good models typically achieve R² > 0.95 for battery voltage estimation.

---

##      Understanding Your Results

### Output Files Structure

Each training session creates a `job_YYYYMMDD-HHMMSS/` folder containing:

```
job_20250711-140144/
├── models/
│   └── model_LSTM_hu_10_layers_1/
│       ├── task_xxx_1_rep_1/
│       │   ├── model.pth              # Trained model
│       │   ├── hyperparams.json       # Model configuration
│       │   └── logs/
│       │       ├── training_progress.csv    # Training metrics
│       │       └── task_xxx_training.db     # Detailed logs
├── scalers/
│   └── augmentation_scaler.joblib     # Data normalization parameters
└── processed_data/                    # Processed datasets
```

### Key Files:
- **`model.pth`** - Your trained model (load this for testing)
- **`hyperparams.json`** - All settings used for this training
- **`training_progress.csv`** - Training loss and validation metrics
- **`augmentation_scaler.joblib`** - Normalization parameters (needed for testing)

---

## 🛠️ Advanced Usage

### Using Your Own Data

1. **Data Format Requirements**:
   - CSV files with headers
   - Required columns: timestamp, voltage, current, temperature, SOC
   - Time-series data (sequential measurements)

2. **Data Preparation**:
   - Ensure consistent sampling rates
   - Remove large gaps in data
   - Check for reasonable value ranges

3. **File Organization**:
   - Place training files in `data/train_data/`
   - Place validation files in `data/val_data/`
   - Place test files in `data/test_data/`

### Model Optimization

1. **If Model Performance is Poor**:
   - Increase hidden units (10 → 50 → 100)
   - Add more layers (1 → 2 → 3)
   - Increase training epochs (5 → 20 → 50)
   - Check data quality and preprocessing

2. **If Training is Too Slow**:
   - Reduce batch size (200 → 100 → 50)
   - Reduce lookback window (400 → 200 → 100)
   - Use fewer hidden units
   - Enable GPU if available

3. **If Model Overfits** (great training, poor testing):
   - Reduce model complexity (fewer units/layers)
   - Increase validation patience
   - Add more training data

### Hyperparameter Tuning Tips

- **Hidden Units**: Start with 10, increase if underfitting
- **Layers**: Usually 1-2 layers sufficient for battery data
- **Lookback**: 400 works well for battery data (adjust based on sampling rate)
- **Learning Rate**: 0.0001 is conservative and stable
- **Batch Size**: Larger = more stable, smaller = faster updates

---

## 🚨 Troubleshooting

### Common Issues and Solutions

#### "No data files found"
- **Solution**: Ensure CSV files are in the correct folders
- Check file format and headers match expected columns

#### "Training fails immediately"
- **Solution**: Check data for NaN values or infinite numbers
- Verify all selected feature columns exist in your data
- Try reducing batch size or model complexity

#### "Poor model performance"
- **Solution**: 
  - Ensure data normalization is enabled
  - Check that voltage values are reasonable
  - Verify train/validation/test splits represent similar conditions
  - Try increasing model complexity gradually

#### "Out of memory errors"
- **Solution**:
  - Reduce batch size (200 → 100 → 50)
  - Reduce lookback window (400 → 200)
  - Use smaller model (fewer hidden units)
  - Close other applications

#### "Training takes too long"
- **Solution**:
  - Reduce max epochs for initial testing
  - Use smaller datasets for experimentation
  - Enable early stopping (patience setting)

---

##   Best Practices

### Data Quality
- Use high-quality, consistent battery test data
- Ensure reasonable sampling rates (1-10 Hz typical)
- Remove obvious outliers and sensor errors
- Use data from similar battery types and conditions

### Model Training
- Start with default parameters
- Train on representative data
- Use separate test data not seen during training
- Save successful model configurations

### Workflow Efficiency
- Use demo data to learn the interface
- Save hyperparameter configurations that work well
- Document successful settings for future use
- Regularly backup important trained models

---

## 📚 Understanding Machine Learning Concepts

### Time Series Prediction
VEstim predicts future voltage values based on historical patterns in:
- State of Charge (SOC)
- Current draw
- Temperature
- Previous voltage values

### Model Types Explained

**LSTM (Long Short-Term Memory)**:
- Best for capturing long-term dependencies
- Handles sequential data very well
- Recommended for battery voltage prediction
- More complex but typically more accurate

**GRU (Gated Recurrent Unit)**:
- Simpler than LSTM, faster training
- Good balance of performance and efficiency
- Alternative when LSTM is too slow

**FNN (Feed-Forward Neural Network)**:
- Simplest model type
- Treats each data point independently
- Fast training and prediction
- Good for non-sequential patterns

### Training Process
1. **Forward Pass**: Model makes predictions
2. **Loss Calculation**: Compare predictions to actual values
3. **Backward Pass**: Adjust model weights to reduce error
4. **Repeat**: Continue until model converges or max epochs reached

### Validation
- Uses separate data to check if model generalizes well
- Prevents overfitting (memorizing training data)
- Automatic early stopping when validation stops improving

---

##     Data Management

### Backup Important Files
- Trained models (`model.pth` files)
- Successful hyperparameter configurations
- Your original datasets
- Custom preprocessing settings

### Organizing Results
- Each training session creates a unique job folder
- Keep successful models and delete failed experiments
- Document what settings worked for different datasets
- Export successful configurations for reuse

---

##    Updating Default Settings

VEstim remembers your preferences:
- **Last used folders** - Automatically loads your recent data folders
- **Last used hyperparameters** - Saves successful model configurations
- **File format preferences** - Remembers CSV/MAT/XLSX choices

Settings are saved in `default_settings.json` - this file contains your personalized defaults.

---

## 📞 Support and Community

### Getting Help
- Check this README for common issues
- Review error messages carefully
- Try with demo data first to isolate problems
- Document successful configurations for reuse

### Sharing Models
- Share hyperparameter configurations (JSON files)
- Share preprocessing settings
- Document data requirements for your models

---

## 🎉 Congratulations!

You now have everything needed to successfully use VEstim for battery voltage estimation. Start with the demo data, follow the 4-step workflow, and gradually experiment with your own datasets and model configurations.

**Remember**: 
- Start simple with default settings
- Use the demo data to learn the interface
- Gradually increase complexity as needed
- Save configurations that work well

Happy modeling! 🔋⚡

---

*VEstim Version 1.0 - For battery voltage estimation using machine learning*
