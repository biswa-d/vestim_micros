# VEstim Model Deployment Guide for BMS Integration

## Overview

This guide provides comprehensive instructions for deploying VEstim-trained models in Battery Management System (BMS) environments, including C/C++ integration, real-time inference, and production optimization.

**Target Audience:** BMS developers, embedded systems engineers, and production deployment teams.

---

## Prerequisites

### Required Files from VEstim Training
After training a model in VEstim, you'll find these essential files in your job output folder:

```
job_YYYYMMDD-HHMMSS/
├── best_model.pth                    # Main model checkpoint
├── best_model_export.pt              # Portable export with metadata
├── BEST_MODEL_LOADING_INSTRUCTIONS.md # Auto-generated loading guide
├── scalers/
│   ├── augmentation_scaler.joblib    # Data normalization scaler
│   └── augmentation_scaler_statistics.txt # Human-readable scaler info
├── hyperparams.json                  # Model configuration
└── training_progress.csv             # Training metrics
```

### Development Environment Requirements
- **Python 3.8+** with PyTorch for model conversion
- **C++ compiler** (GCC, MSVC, or Clang) for production deployment
- **LibTorch** (PyTorch C++ API) for native C++ inference
- **ONNX Runtime** (optional) for cross-platform deployment

---

## Deployment Options

### Option 1: Python Integration (Fastest Setup)

**Best for:** Rapid prototyping, testing, Python-based BMS systems

```python
import torch
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class VEstimBMSModel:
    def __init__(self, model_path, scaler_path):
        """
        Initialize the VEstim model for BMS deployment
        
        Args:
            model_path: Path to best_model_export.pt
            scaler_path: Path to augmentation_scaler.joblib
        """
        # Load model checkpoint
        self.checkpoint = torch.load(model_path, map_location='cpu')
        self.scaler = joblib.load(scaler_path)
        
        # Extract model metadata
        self.model_type = self.checkpoint['model_type']
        self.input_size = self.checkpoint['hyperparams']['input_size']
        self.lookback = self.checkpoint['data_config']['lookback']
        self.feature_columns = self.checkpoint['data_config']['feature_columns']
        
        # Initialize model
        self._create_model()
        
        # Performance optimization
        self.model.eval()
        torch.set_num_threads(1)  # Optimize for single-threaded inference
    
    def _create_model(self):
        """Create and load the PyTorch model"""
        if self.model_type == 'LSTM':
            from vestim_model_definitions import LSTMModel
            self.model = LSTMModel(
                input_size=self.checkpoint['hyperparams']['input_size'],
                hidden_units=self.checkpoint['hyperparams']['hidden_size'],
                num_layers=self.checkpoint['hyperparams']['num_layers'],
                output_size=self.checkpoint['hyperparams']['output_size']
            )
        elif self.model_type == 'FNN':
            from vestim_model_definitions import FNNModel
            self.model = FNNModel(
                input_size=self.checkpoint['hyperparams']['input_size'],
                hidden_layer_sizes=self.checkpoint['hyperparams']['hidden_layer_sizes'],
                output_size=self.checkpoint['hyperparams']['output_size'],
                dropout_prob=0.0  # No dropout during inference
            )
        
        self.model.load_state_dict(self.checkpoint['state_dict'])
    
    def predict_voltage(self, soc, current, temperature):
        """
        Predict battery voltage for BMS integration
        
        Args:
            soc: State of charge (0.0 to 1.0)
            current: Battery current in Amperes
            temperature: Temperature in Celsius
            
        Returns:
            predicted_voltage: Estimated voltage in Volts
        """
        # Prepare input data
        raw_data = np.array([[soc, current, temperature]])
        
        # Normalize inputs using training scaler
        normalized_data = self.scaler.transform(raw_data)
        
        # Create sequence for temporal models
        if self.model_type in ['LSTM', 'GRU']:
            # For first prediction, repeat the current sample
            sequence = np.tile(normalized_data, (self.lookback, 1))
            input_tensor = torch.FloatTensor(sequence).unsqueeze(0)  # Shape: (1, lookback, features)
        else:  # FNN
            input_tensor = torch.FloatTensor(normalized_data)  # Shape: (1, features)
        
        # Model inference
        with torch.no_grad():
            if self.model_type in ['LSTM', 'GRU']:
                output, _ = self.model(input_tensor)
            else:
                output = self.model(input_tensor)
        
        # Denormalize output to get voltage
        voltage_normalized = output.numpy()
        voltage = self._denormalize_voltage(voltage_normalized)
        
        return float(voltage[0, 0])
    
    def _denormalize_voltage(self, normalized_voltage):
        """Convert normalized voltage back to real voltage"""
        # Assuming voltage is the target column (last column in scaler)
        voltage_min = self.scaler.data_min_[-1]
        voltage_max = self.scaler.data_max_[-1]
        return normalized_voltage * (voltage_max - voltage_min) + voltage_min

# Usage example
model = VEstimBMSModel('best_model_export.pt', 'augmentation_scaler.joblib')
voltage = model.predict_voltage(soc=0.8, current=-10.5, temperature=25.0)
print(f"Predicted voltage: {voltage:.3f}V")
```

### Option 2: C++ Integration (Production Ready)

**Best for:** Real-time embedded systems, high-performance BMS, minimal latency

#### Step 1: Export Model to TorchScript

```python
# export_to_torchscript.py
import torch
import joblib
import numpy as np

def export_vestim_model_cpp(model_path, scaler_path, output_path):
    """Export VEstim model for C++ deployment"""
    
    # Load checkpoint
    checkpoint = torch.load(model_path)
    
    # Recreate model (using definitions from checkpoint)
    exec(checkpoint['model_definition'])
    
    if checkpoint['model_type'] == 'LSTM':
        model = LSTMModel(
            input_size=checkpoint['hyperparams']['input_size'],
            hidden_units=checkpoint['hyperparams']['hidden_size'],
            num_layers=checkpoint['hyperparams']['num_layers'],
            output_size=checkpoint['hyperparams']['output_size']
        )
    
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # Create example input for tracing
    lookback = checkpoint['data_config']['lookback']
    input_size = checkpoint['hyperparams']['input_size']
    example_input = torch.randn(1, lookback, input_size)
    
    # Trace the model
    traced_model = torch.jit.trace(model, example_input)
    
    # Save traced model
    traced_model.save(output_path)
    
    # Export scaler parameters for C++
    scaler = joblib.load(scaler_path)
    scaler_params = {
        'data_min': scaler.data_min_.tolist(),
        'data_max': scaler.data_max_.tolist(),
        'feature_names': checkpoint['data_config']['feature_columns']
    }
    
    return scaler_params

# Export model
scaler_params = export_vestim_model_cpp(
    'best_model_export.pt', 
    'augmentation_scaler.joblib', 
    'vestim_model_traced.pt'
)
print("Model exported for C++ deployment")
```

#### Step 2: C++ Implementation

```cpp
// vestim_bms_model.hpp
#pragma once
#include <torch/script.h>
#include <vector>
#include <memory>

class VEstimBMSModel {
private:
    torch::jit::script::Module model_;
    std::vector<float> data_min_;
    std::vector<float> data_max_;
    int lookback_;
    int input_size_;
    
public:
    VEstimBMSModel(const std::string& model_path, 
                   const std::vector<float>& data_min,
                   const std::vector<float>& data_max,
                   int lookback, int input_size);
    
    float predictVoltage(float soc, float current, float temperature);
    
private:
    std::vector<float> normalize(const std::vector<float>& raw_data);
    float denormalizeVoltage(float normalized_voltage);
};

// vestim_bms_model.cpp
#include "vestim_bms_model.hpp"
#include <algorithm>

VEstimBMSModel::VEstimBMSModel(const std::string& model_path,
                               const std::vector<float>& data_min,
                               const std::vector<float>& data_max,
                               int lookback, int input_size)
    : data_min_(data_min), data_max_(data_max), 
      lookback_(lookback), input_size_(input_size) {
    
    try {
        model_ = torch::jit::load(model_path);
        model_.eval();
        
        // Optimize for inference
        torch::set_num_threads(1);
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load model: " + std::string(e.what()));
    }
}

float VEstimBMSModel::predictVoltage(float soc, float current, float temperature) {
    // Prepare input data
    std::vector<float> raw_data = {soc, current, temperature};
    std::vector<float> normalized = normalize(raw_data);
    
    // Create input tensor
    torch::Tensor input_tensor = torch::zeros({1, lookback_, input_size_});
    
    // Fill tensor with repeated normalized data for sequence models
    for (int t = 0; t < lookback_; ++t) {
        for (int f = 0; f < input_size_; ++f) {
            input_tensor[0][t][f] = normalized[f];
        }
    }
    
    // Model inference
    std::vector<torch::jit::IValue> inputs = {input_tensor};
    torch::Tensor output = model_.forward(inputs).toTensor();
    
    // Extract and denormalize voltage
    float normalized_voltage = output[0][0].item<float>();
    return denormalizeVoltage(normalized_voltage);
}

std::vector<float> VEstimBMSModel::normalize(const std::vector<float>& raw_data) {
    std::vector<float> normalized(raw_data.size());
    for (size_t i = 0; i < raw_data.size(); ++i) {
        normalized[i] = (raw_data[i] - data_min_[i]) / (data_max_[i] - data_min_[i]);
    }
    return normalized;
}

float VEstimBMSModel::denormalizeVoltage(float normalized_voltage) {
    // Voltage is typically the last feature in the scaler
    int voltage_idx = data_min_.size() - 1;
    return normalized_voltage * (data_max_[voltage_idx] - data_min_[voltage_idx]) + data_min_[voltage_idx];
}

// Example usage in BMS main loop
int main() {
    // Initialize model with exported parameters
    std::vector<float> data_min = {0.0, -50.0, -10.0, 2.5};  // From Python export
    std::vector<float> data_max = {1.0, 50.0, 60.0, 4.2};    // From Python export
    
    VEstimBMSModel model("vestim_model_traced.pt", data_min, data_max, 10, 3);
    
    // BMS sensor readings
    float soc = 0.8f;
    float current = -10.5f;
    float temperature = 25.0f;
    
    // Predict voltage
    float predicted_voltage = model.predictVoltage(soc, current, temperature);
    
    printf("Predicted voltage: %.3f V\n", predicted_voltage);
    
    return 0;
}
```

#### Step 3: CMake Build Configuration

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.12)
project(VEstimBMS)

find_package(Torch REQUIRED)

# Create executable
add_executable(vestim_bms
    vestim_bms_model.cpp
    main.cpp
)

# Link libraries
target_link_libraries(vestim_bms ${TORCH_LIBRARIES})

# Set C++ standard
set_property(TARGET vestim_bms PROPERTY CXX_STANDARD 14)

# Optimization flags for production
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(vestim_bms PRIVATE -O3 -march=native)
endif()
```

### Option 3: ONNX Runtime (Cross-Platform)

**Best for:** Cross-platform deployment, standardized inference runtime

```python
# export_to_onnx.py
import torch
import torch.onnx

def export_vestim_to_onnx(model_path, output_path):
    """Export VEstim model to ONNX format"""
    
    checkpoint = torch.load(model_path)
    
    # Recreate model
    exec(checkpoint['model_definition'])
    
    if checkpoint['model_type'] == 'LSTM':
        model = LSTMModel(
            input_size=checkpoint['hyperparams']['input_size'],
            hidden_units=checkpoint['hyperparams']['hidden_size'],
            num_layers=checkpoint['hyperparams']['num_layers'],
            output_size=checkpoint['hyperparams']['output_size']
        )
    
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # Create dummy input
    lookback = checkpoint['data_config']['lookback']
    input_size = checkpoint['hyperparams']['input_size']
    dummy_input = torch.randn(1, lookback, input_size)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['voltage'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'voltage': {0: 'batch_size'}
        }
    )

export_vestim_to_onnx('best_model_export.pt', 'vestim_model.onnx')
```

```cpp
// onnx_inference.cpp - Using ONNX Runtime
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <memory>

class VEstimONNXModel {
private:
    std::unique_ptr<Ort::Session> session_;
    Ort::Env env_;
    std::vector<float> data_min_, data_max_;
    
public:
    VEstimONNXModel(const std::string& model_path,
                    const std::vector<float>& data_min,
                    const std::vector<float>& data_max) 
        : env_(ORT_LOGGING_LEVEL_WARNING, "VEstimBMS"),
          data_min_(data_min), data_max_(data_max) {
        
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);
    }
    
    float predictVoltage(float soc, float current, float temperature) {
        // Normalize inputs
        std::vector<float> normalized = {
            (soc - data_min_[0]) / (data_max_[0] - data_min_[0]),
            (current - data_min_[1]) / (data_max_[1] - data_min_[1]),
            (temperature - data_min_[2]) / (data_max_[2] - data_min_[2])
        };
        
        // Prepare input tensor (assuming lookback=10, input_size=3)
        std::vector<float> input_data(1 * 10 * 3);
        for (int t = 0; t < 10; ++t) {
            for (int f = 0; f < 3; ++f) {
                input_data[t * 3 + f] = normalized[f];
            }
        }
        
        // Create input tensor
        std::vector<int64_t> input_shape = {1, 10, 3};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            env_.GetAllocatorWithDefaultOptions(), input_data.data(), 
            input_data.size(), input_shape.data(), input_shape.size());
        
        // Run inference
        std::vector<const char*> input_names = {"input"};
        std::vector<const char*> output_names = {"voltage"};
        
        auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, 
                                          input_names.data(), &input_tensor, 1,
                                          output_names.data(), 1);
        
        // Extract result
        float* output_data = output_tensors.front().GetTensorMutableData<float>();
        
        // Denormalize voltage
        int voltage_idx = data_min_.size() - 1;
        return output_data[0] * (data_max_[voltage_idx] - data_min_[voltage_idx]) + data_min_[voltage_idx];
    }
};
```

---

## Data Preparation for Production

### Input Data Requirements

```python
def prepare_bms_data(soc, current, temperature, sequence_buffer, lookback):
    """
    Prepare real-time BMS data for model inference
    
    Args:
        soc: Current state of charge (0.0 to 1.0)
        current: Battery current in Amperes (positive = charging)
        temperature: Temperature in Celsius
        sequence_buffer: Circular buffer of recent measurements
        lookback: Model sequence length requirement
    
    Returns:
        formatted_input: Ready for model inference
    """
    
    # Add current measurement to buffer
    current_sample = [soc, current, temperature]
    sequence_buffer.append(current_sample)
    
    # Maintain buffer size
    if len(sequence_buffer) > lookback:
        sequence_buffer.pop(0)
    
    # Handle insufficient history (cold start)
    if len(sequence_buffer) < lookback:
        # Repeat current sample to fill sequence
        padded_sequence = [current_sample] * lookback
    else:
        padded_sequence = list(sequence_buffer)
    
    return np.array(padded_sequence)
```

### Data Quality Validation

```python
def validate_bms_inputs(soc, current, temperature):
    """Validate BMS sensor inputs before inference"""
    
    errors = []
    
    # SOC validation
    if not (0.0 <= soc <= 1.0):
        errors.append(f"SOC out of range: {soc} (expected 0.0-1.0)")
    
    # Current validation (typical range for automotive batteries)
    if not (-200.0 <= current <= 200.0):
        errors.append(f"Current out of range: {current}A (expected -200 to +200A)")
    
    # Temperature validation
    if not (-40.0 <= temperature <= 80.0):
        errors.append(f"Temperature out of range: {temperature}°C (expected -40 to +80°C)")
    
    # Check for NaN or infinite values
    for name, value in [("SOC", soc), ("Current", current), ("Temperature", temperature)]:
        if not np.isfinite(value):
            errors.append(f"{name} is not finite: {value}")
    
    return errors

# Usage in BMS loop
errors = validate_bms_inputs(soc, current, temperature)
if errors:
    print("Input validation failed:", errors)
    # Use fallback estimation or previous prediction
else:
    voltage = model.predict_voltage(soc, current, temperature)
```

---

## Performance Optimization

### Inference Speed Optimization

```python
# Python optimization techniques
class OptimizedVEstimModel:
    def __init__(self, model_path, scaler_path):
        self.model = self._load_optimized_model(model_path)
        self.scaler_params = self._extract_scaler_params(scaler_path)
        
        # Pre-allocate tensors to avoid memory allocation overhead
        self.input_tensor = torch.zeros(1, self.lookback, self.input_size)
        
    def _load_optimized_model(self, model_path):
        """Load model with optimization flags"""
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Create model and load weights
        model = self._create_model(checkpoint)
        model.eval()
        
        # Apply optimizations
        torch.set_num_threads(1)  # Single-threaded for real-time
        model = torch.jit.script(model)  # JIT compilation
        
        # Warm up the model
        dummy_input = torch.randn(1, self.lookback, self.input_size)
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        return model
    
    def fast_predict(self, soc, current, temperature):
        """Optimized prediction with minimal overhead"""
        
        # In-place normalization to avoid array creation
        self.input_tensor[0, :, 0] = (soc - self.scaler_params['min'][0]) / self.scaler_params['range'][0]
        self.input_tensor[0, :, 1] = (current - self.scaler_params['min'][1]) / self.scaler_params['range'][1]
        self.input_tensor[0, :, 2] = (temperature - self.scaler_params['min'][2]) / self.scaler_params['range'][2]
        
        # Fast inference
        with torch.no_grad():
            output = self.model(self.input_tensor)
        
        # Quick denormalization
        voltage_range = self.scaler_params['range'][-1]
        voltage_min = self.scaler_params['min'][-1]
        
        return float(output[0, 0] * voltage_range + voltage_min)
```

### Memory Management

```cpp
// C++ memory optimization
class OptimizedVEstimModel {
private:
    torch::jit::script::Module model_;
    torch::Tensor input_tensor_;  // Pre-allocated
    std::vector<torch::jit::IValue> inputs_;  // Pre-allocated
    
public:
    OptimizedVEstimModel(const std::string& model_path, /* other params */) {
        // Load model
        model_ = torch::jit::load(model_path);
        model_.eval();
        
        // Pre-allocate tensors
        input_tensor_ = torch::zeros({1, lookback_, input_size_});
        inputs_.reserve(1);
        inputs_.push_back(input_tensor_);
        
        // Warm up
        for (int i = 0; i < 10; ++i) {
            model_.forward(inputs_);
        }
    }
    
    float fastPredict(float soc, float current, float temperature) {
        // Reuse pre-allocated tensor (no memory allocation)
        auto accessor = input_tensor_.accessor<float, 3>();
        
        // Normalize and fill tensor
        float norm_soc = (soc - data_min_[0]) / (data_max_[0] - data_min_[0]);
        float norm_current = (current - data_min_[1]) / (data_max_[1] - data_min_[1]);
        float norm_temp = (temperature - data_min_[2]) / (data_max_[2] - data_min_[2]);
        
        for (int t = 0; t < lookback_; ++t) {
            accessor[0][t][0] = norm_soc;
            accessor[0][t][1] = norm_current;
            accessor[0][t][2] = norm_temp;
        }
        
        // Fast inference
        torch::NoGradGuard no_grad;
        torch::Tensor output = model_.forward(inputs_).toTensor();
        
        // Denormalize and return
        return output[0][0].item<float>() * 
               (data_max_[3] - data_min_[3]) + data_min_[3];
    }
};
```

### Benchmarking and Profiling

```python
import time
import numpy as np

def benchmark_model(model, num_iterations=1000):
    """Benchmark model inference performance"""
    
    # Warm up
    for _ in range(10):
        _ = model.predict_voltage(0.5, -10.0, 25.0)
    
    # Benchmark
    start_time = time.perf_counter()
    
    for i in range(num_iterations):
        soc = 0.3 + 0.4 * np.random.random()
        current = -50 + 100 * np.random.random()
        temperature = 10 + 30 * np.random.random()
        
        voltage = model.predict_voltage(soc, current, temperature)
    
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_time_ms = (total_time / num_iterations) * 1000
    
    print(f"Benchmark Results:")
    print(f"  Total time: {total_time:.3f} seconds")
    print(f"  Average inference time: {avg_time_ms:.3f} ms")
    print(f"  Throughput: {num_iterations / total_time:.1f} predictions/second")
    
    return avg_time_ms

# Run benchmark
model = VEstimBMSModel('best_model_export.pt', 'augmentation_scaler.joblib')
avg_time = benchmark_model(model)

# Performance targets for different deployment scenarios
if avg_time < 1.0:
    print("Excellent performance - suitable for high-frequency BMS (>1kHz)")
elif avg_time < 10.0:
    print("Good performance - suitable for standard BMS (>100Hz)")
elif avg_time < 100.0:
    print("Moderate performance - suitable for low-frequency BMS (>10Hz)")
else:
    print("Poor performance - optimization required")
```

---

## Integration Examples

### Real-Time BMS Integration

```cpp
// Example: Integration with automotive BMS
#include "vestim_bms_model.hpp"
#include <thread>
#include <chrono>
#include <atomic>

class BatteryManagementSystem {
private:
    VEstimBMSModel voltage_estimator_;
    std::atomic<bool> running_;
    
    // Sensor data
    std::atomic<float> current_soc_;
    std::atomic<float> current_current_;
    std::atomic<float> current_temperature_;
    std::atomic<float> estimated_voltage_;
    
public:
    BatteryManagementSystem(const std::string& model_path,
                           const std::vector<float>& data_min,
                           const std::vector<float>& data_max,
                           int lookback, int input_size)
        : voltage_estimator_(model_path, data_min, data_max, lookback, input_size),
          running_(true) {}
    
    void start() {
        // Start estimation thread
        std::thread estimation_thread(&BatteryManagementSystem::estimationLoop, this);
        estimation_thread.detach();
    }
    
    void stop() {
        running_ = false;
    }
    
    void updateSensorData(float soc, float current, float temperature) {
        current_soc_ = soc;
        current_current_ = current;
        current_temperature_ = temperature;
    }
    
    float getEstimatedVoltage() const {
        return estimated_voltage_;
    }
    
private:
    void estimationLoop() {
        while (running_) {
            try {
                // Read current sensor values
                float soc = current_soc_;
                float current = current_current_;
                float temperature = current_temperature_;
                
                // Predict voltage
                float voltage = voltage_estimator_.predictVoltage(soc, current, temperature);
                estimated_voltage_ = voltage;
                
                // Log for debugging (remove in production)
                printf("SOC: %.3f, Current: %.1fA, Temp: %.1f°C, Voltage: %.3fV\n",
                       soc, current, temperature, voltage);
                
            } catch (const std::exception& e) {
                printf("Estimation error: %s\n", e.what());
            }
            
            // Sleep for 10ms (100Hz update rate)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
};

// Usage in main BMS application
int main() {
    // Initialize with model parameters (from Python export)
    std::vector<float> data_min = {0.0, -100.0, -20.0, 2.5};
    std::vector<float> data_max = {1.0, 100.0, 60.0, 4.2};
    
    BatteryManagementSystem bms("vestim_model_traced.pt", data_min, data_max, 10, 3);
    bms.start();
    
    // Simulate BMS operation
    for (int i = 0; i < 1000; ++i) {
        // Simulate sensor readings
        float soc = 0.2 + 0.6 * (float)i / 1000.0;  // SOC increasing
        float current = -20.0 + 5.0 * sin(i * 0.1);  // Varying current
        float temperature = 25.0 + 5.0 * sin(i * 0.05);  // Varying temperature
        
        bms.updateSensorData(soc, current, temperature);
        
        // Get estimated voltage
        float voltage = bms.getEstimatedVoltage();
        
        // Use voltage estimate in BMS logic
        // ... (safety checks, charge control, etc.)
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    bms.stop();
    return 0;
}
```

### Embedded System Integration

```c
// Example: Microcontroller integration (STM32, Arduino, etc.)
// Note: This requires a lightweight inference engine like TensorFlow Lite Micro

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

class EmbeddedVEstimModel {
private:
    tflite::MicroErrorReporter error_reporter_;
    const tflite::Model* model_;
    tflite::MicroInterpreter* interpreter_;
    TfLiteTensor* input_;
    TfLiteTensor* output_;
    
    // Scaler parameters (stored in flash)
    const float data_min_[4] = {0.0f, -50.0f, -10.0f, 2.5f};
    const float data_max_[4] = {1.0f, 50.0f, 60.0f, 4.2f};
    
    // Working memory for TensorFlow Lite Micro
    uint8_t tensor_arena_[8192];  // Adjust size based on model
    
public:
    bool initialize(const unsigned char* model_data, int model_size) {
        // Load model
        model_ = tflite::GetModel(model_data);
        if (model_->version() != TFLITE_SCHEMA_VERSION) {
            return false;
        }
        
        // Create resolver and interpreter
        static tflite::AllOpsResolver resolver;
        static tflite::MicroInterpreter static_interpreter(
            model_, resolver, tensor_arena_, sizeof(tensor_arena_), &error_reporter_);
        interpreter_ = &static_interpreter;
        
        // Allocate tensors
        if (interpreter_->AllocateTensors() != kTfLiteOk) {
            return false;
        }
        
        // Get input and output tensors
        input_ = interpreter_->input(0);
        output_ = interpreter_->output(0);
        
        return true;
    }
    
    float predictVoltage(float soc, float current, float temperature) {
        // Normalize inputs
        float norm_soc = (soc - data_min_[0]) / (data_max_[0] - data_min_[0]);
        float norm_current = (current - data_min_[1]) / (data_max_[1] - data_min_[1]);
        float norm_temp = (temperature - data_min_[2]) / (data_max_[2] - data_min_[2]);
        
        // Fill input tensor (assuming FNN model for simplicity)
        input_->data.f[0] = norm_soc;
        input_->data.f[1] = norm_current;
        input_->data.f[2] = norm_temp;
        
        // Run inference
        if (interpreter_->Invoke() != kTfLiteOk) {
            return -1.0f;  // Error indicator
        }
        
        // Get output and denormalize
        float norm_voltage = output_->data.f[0];
        return norm_voltage * (data_max_[3] - data_min_[3]) + data_min_[3];
    }
};

// Usage in embedded application
EmbeddedVEstimModel model;

void setup() {
    // Initialize model with converted TensorFlow Lite model
    extern const unsigned char vestim_model_tflite[];
    extern const int vestim_model_tflite_len;
    
    if (model.initialize(vestim_model_tflite, vestim_model_tflite_len)) {
        Serial.println("VEstim model initialized successfully");
    } else {
        Serial.println("Failed to initialize VEstim model");
    }
}

void loop() {
    // Read sensors
    float soc = readSOC();
    float current = readCurrent();
    float temperature = readTemperature();
    
    // Predict voltage
    float voltage = model.predictVoltage(soc, current, temperature);
    
    // Use in control logic
    updateBMSControl(voltage);
    
    delay(10);  // 100Hz update rate
}
```

---

## Production Considerations

### Error Handling and Failsafes

```python
class RobustVEstimModel:
    def __init__(self, model_path, scaler_path):
        self.model = None
        self.scaler = None
        self.last_valid_prediction = None
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
        
        try:
            self._initialize_model(model_path, scaler_path)
        except Exception as e:
            print(f"Model initialization failed: {e}")
            raise
    
    def predict_voltage_safe(self, soc, current, temperature):
        """Thread-safe prediction with error handling"""
        try:
            # Validate inputs
            errors = validate_bms_inputs(soc, current, temperature)
            if errors:
                raise ValueError(f"Input validation failed: {errors}")
            
            # Make prediction
            voltage = self._predict_internal(soc, current, temperature)
            
            # Sanity check output
            if not (2.0 <= voltage <= 5.0):  # Reasonable voltage range
                raise ValueError(f"Voltage prediction out of range: {voltage}V")
            
            # Reset error counter and update last valid prediction
            self.consecutive_errors = 0
            self.last_valid_prediction = voltage
            
            return voltage
            
        except Exception as e:
            self.consecutive_errors += 1
            
            if self.consecutive_errors > self.max_consecutive_errors:
                raise RuntimeError(f"Too many consecutive prediction errors: {e}")
            
            # Return fallback prediction
            return self._get_fallback_prediction(soc, current, temperature)
    
    def _get_fallback_prediction(self, soc, current, temperature):
        """Fallback voltage estimation when model fails"""
        if self.last_valid_prediction is not None:
            return self.last_valid_prediction
        
        # Simple voltage model as fallback
        # V = V_nominal + K_soc * (SOC - 0.5) + K_temp * (T - 25)
        V_nominal = 3.7  # Nominal voltage
        K_soc = 0.8      # SOC coefficient
        K_temp = -0.002  # Temperature coefficient
        
        voltage = V_nominal + K_soc * (soc - 0.5) + K_temp * (temperature - 25.0)
        return max(2.5, min(4.2, voltage))  # Clamp to reasonable range
```

### Model Versioning and Updates

```python
class VersionedVEstimModel:
    def __init__(self, model_directory):
        self.model_directory = model_directory
        self.current_model = None
        self.model_version = None
        
    def load_latest_model(self):
        """Load the most recent model version"""
        model_files = glob.glob(os.path.join(self.model_directory, "vestim_model_v*.pt"))
        
        if not model_files:
            raise FileNotFoundError("No model files found")
        
        # Sort by version number
        model_files.sort(key=lambda x: int(re.search(r'v(\d+)', x).group(1)))
        latest_model = model_files[-1]
        
        # Load model
        self.current_model = VEstimBMSModel(latest_model, 
                                          os.path.join(self.model_directory, "scaler.joblib"))
        
        # Extract version
        self.model_version = re.search(r'v(\d+)', latest_model).group(1)
        
        print(f"Loaded VEstim model version: {self.model_version}")
    
    def check_for_updates(self):
        """Check if a newer model version is available"""
        try:
            old_version = int(self.model_version) if self.model_version else 0
            
            model_files = glob.glob(os.path.join(self.model_directory, "vestim_model_v*.pt"))
            if model_files:
                latest_file = max(model_files, key=lambda x: int(re.search(r'v(\d+)', x).group(1)))
                latest_version = int(re.search(r'v(\d+)', latest_file).group(1))
                
                if latest_version > old_version:
                    print(f"New model version available: v{latest_version}")
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error checking for updates: {e}")
            return False
    
    def update_model(self):
        """Hot-swap to newer model version"""
        try:
            old_model = self.current_model
            old_version = self.model_version
            
            self.load_latest_model()
            
            print(f"Model updated from v{old_version} to v{self.model_version}")
            return True
            
        except Exception as e:
            print(f"Model update failed: {e}")
            # Restore old model
            self.current_model = old_model
            self.model_version = old_version
            return False
```

### Monitoring and Diagnostics

```python
class VEstimModelMonitor:
    def __init__(self, model):
        self.model = model
        self.predictions = []
        self.timestamps = []
        self.performance_metrics = {
            'avg_inference_time': 0.0,
            'max_inference_time': 0.0,
            'total_predictions': 0,
            'error_count': 0
        }
    
    def predict_with_monitoring(self, soc, current, temperature):
        """Prediction with performance monitoring"""
        start_time = time.perf_counter()
        
        try:
            voltage = self.model.predict_voltage(soc, current, temperature)
            
            # Record successful prediction
            end_time = time.perf_counter()
            inference_time = end_time - start_time
            
            self._update_performance_metrics(inference_time, success=True)
            self._record_prediction(soc, current, temperature, voltage, inference_time)
            
            return voltage
            
        except Exception as e:
            self._update_performance_metrics(0, success=False)
            raise
    
    def _update_performance_metrics(self, inference_time, success):
        """Update performance statistics"""
        if success:
            self.performance_metrics['total_predictions'] += 1
            
            # Update average inference time
            total = self.performance_metrics['total_predictions']
            current_avg = self.performance_metrics['avg_inference_time']
            self.performance_metrics['avg_inference_time'] = \
                (current_avg * (total - 1) + inference_time) / total
            
            # Update max inference time
            self.performance_metrics['max_inference_time'] = \
                max(self.performance_metrics['max_inference_time'], inference_time)
        else:
            self.performance_metrics['error_count'] += 1
    
    def _record_prediction(self, soc, current, temp, voltage, inference_time):
        """Record prediction for analysis"""
        self.predictions.append({
            'timestamp': time.time(),
            'soc': soc,
            'current': current,
            'temperature': temp,
            'voltage': voltage,
            'inference_time': inference_time
        })
        
        # Keep only recent predictions (memory management)
        if len(self.predictions) > 10000:
            self.predictions = self.predictions[-5000:]
    
    def get_diagnostics(self):
        """Get current diagnostic information"""
        error_rate = (self.performance_metrics['error_count'] / 
                     max(1, self.performance_metrics['total_predictions'] + 
                         self.performance_metrics['error_count'])) * 100
        
        return {
            'model_status': 'healthy' if error_rate < 1.0 else 'degraded',
            'total_predictions': self.performance_metrics['total_predictions'],
            'error_rate_percent': error_rate,
            'avg_inference_time_ms': self.performance_metrics['avg_inference_time'] * 1000,
            'max_inference_time_ms': self.performance_metrics['max_inference_time'] * 1000,
            'recent_predictions': len(self.predictions)
        }
    
    def export_diagnostics(self, filepath):
        """Export diagnostics to file for analysis"""
        diagnostics = self.get_diagnostics()
        
        with open(filepath, 'w') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'diagnostics': diagnostics,
                'recent_predictions': self.predictions[-100:]  # Last 100 predictions
            }, f, indent=2)
```

---

## Testing and Validation

### Unit Testing

```python
import unittest
import numpy as np

class TestVEstimBMSModel(unittest.TestCase):
    def setUp(self):
        self.model = VEstimBMSModel('best_model_export.pt', 'augmentation_scaler.joblib')
    
    def test_basic_prediction(self):
        """Test basic voltage prediction"""
        voltage = self.model.predict_voltage(0.5, -10.0, 25.0)
        self.assertIsInstance(voltage, float)
        self.assertGreater(voltage, 2.0)
        self.assertLess(voltage, 5.0)
    
    def test_input_validation(self):
        """Test input validation"""
        # SOC out of range
        with self.assertRaises(ValueError):
            validate_bms_inputs(1.5, -10.0, 25.0)
        
        # Current out of range
        with self.assertRaises(ValueError):
            validate_bms_inputs(0.5, -500.0, 25.0)
        
        # Temperature out of range
        with self.assertRaises(ValueError):
            validate_bms_inputs(0.5, -10.0, 100.0)
    
    def test_prediction_consistency(self):
        """Test that identical inputs produce identical outputs"""
        voltage1 = self.model.predict_voltage(0.7, -15.0, 30.0)
        voltage2 = self.model.predict_voltage(0.7, -15.0, 30.0)
        self.assertAlmostEqual(voltage1, voltage2, places=6)
    
    def test_performance_requirements(self):
        """Test inference speed requirements"""
        start_time = time.perf_counter()
        
        for _ in range(100):
            self.model.predict_voltage(0.5, -10.0, 25.0)
        
        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / 100
        
        # Should complete in less than 10ms per prediction
        self.assertLess(avg_time, 0.01)

if __name__ == '__main__':
    unittest.main()
```

### Integration Testing

```python
def test_bms_integration():
    """Test full BMS integration scenario"""
    
    # Initialize model
    model = VEstimBMSModel('best_model_export.pt', 'augmentation_scaler.joblib')
    monitor = VEstimModelMonitor(model)
    
    # Simulate BMS operation over time
    test_scenarios = [
        # (soc, current, temperature, expected_voltage_range)
        (0.2, -50.0, 25.0, (2.8, 3.2)),  # Low SOC, high discharge
        (0.8, 20.0, 25.0, (3.8, 4.1)),   # High SOC, charging
        (0.5, 0.0, 25.0, (3.6, 3.8)),    # Mid SOC, no current
        (0.5, -10.0, 0.0, (3.5, 3.7)),   # Cold temperature
        (0.5, -10.0, 50.0, (3.6, 3.8)),  # Hot temperature
    ]
    
    results = []
    for soc, current, temp, expected_range in test_scenarios:
        voltage = monitor.predict_with_monitoring(soc, current, temp)
        
        # Validate prediction is in expected range
        assert expected_range[0] <= voltage <= expected_range[1], \
            f"Voltage {voltage}V not in expected range {expected_range} for SOC={soc}, I={current}, T={temp}"
        
        results.append({
            'inputs': (soc, current, temp),
            'voltage': voltage,
            'expected_range': expected_range,
            'pass': True
        })
    
    # Check diagnostics
    diagnostics = monitor.get_diagnostics()
    assert diagnostics['model_status'] == 'healthy'
    assert diagnostics['error_rate_percent'] == 0.0
    
    print("All BMS integration tests passed")
    return results

# Run integration test
test_results = test_bms_integration()
```

---

## Deployment Checklist

### Pre-Deployment Validation

- [ ] **Model Files Ready**
  - [ ] `best_model_export.pt` available
  - [ ] `augmentation_scaler.joblib` available
  - [ ] Model metadata verified
  
- [ ] **Performance Testing**
  - [ ] Inference time < 10ms (or your requirement)
  - [ ] Memory usage within limits
  - [ ] CPU usage acceptable
  
- [ ] **Accuracy Validation**
  - [ ] Model tested on representative data
  - [ ] Voltage predictions within ±50mV typical
  - [ ] Edge cases handled properly
  
- [ ] **Integration Testing**
  - [ ] BMS integration working
  - [ ] Error handling verified
  - [ ] Failsafe mechanisms tested

### Production Deployment

- [ ] **Environment Setup**
  - [ ] LibTorch/ONNX Runtime installed
  - [ ] Model files deployed securely
  - [ ] Configuration files updated
  
- [ ] **Monitoring Setup**
  - [ ] Performance monitoring enabled
  - [ ] Error logging configured
  - [ ] Diagnostics collection active
  
- [ ] **Safety Measures**
  - [ ] Fallback estimation implemented
  - [ ] Input validation active
  - [ ] Output range checking enabled

### Post-Deployment

- [ ] **Monitoring**
  - [ ] Track prediction accuracy
  - [ ] Monitor inference performance
  - [ ] Watch for error patterns
  
- [ ] **Maintenance**
  - [ ] Regular model updates planned
  - [ ] Performance optimization ongoing
  - [ ] Documentation kept current

---

## Support and Troubleshooting

### Common Issues and Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| High inference time | >50ms per prediction | Use TorchScript, reduce model size, optimize hardware |
| Memory leaks | Increasing RAM usage | Pre-allocate tensors, use proper memory management |
| Accuracy degradation | Predictions drift over time | Validate input data quality, retrain model |
| Model loading failures | Runtime errors | Check file paths, verify model compatibility |

### Performance Optimization Tips

1. **Use TorchScript** for production deployment
2. **Pre-allocate tensors** to avoid memory allocation overhead
3. **Set thread count** to 1 for single-threaded inference
4. **Quantize models** for embedded deployment
5. **Profile your code** to identify bottlenecks

### Getting Help

- **VEstim Documentation**: Check packaging/ folder for guides
- **PyTorch Documentation**: https://pytorch.org/docs/stable/
- **ONNX Runtime**: https://onnxruntime.ai/docs/
- **LibTorch**: https://pytorch.org/cppdocs/

---

## License and Legal

This deployment guide is part of the VEstim project. Please ensure compliance with all applicable licenses when deploying models in production environments.

**Important**: Always validate model predictions against known-good data before deploying in safety-critical applications.

---

*Generated: September 16, 2025*
*Version: 1.0*