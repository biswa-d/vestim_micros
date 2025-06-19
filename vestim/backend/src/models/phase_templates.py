"""
Phase Data Templates for JobContainer State Management

This module defines the data structures required for each phase of the ML pipeline.
Each template defines the minimum data needed for a GUI to restore its state.
"""

import copy
from typing import Dict, Any, List
from datetime import datetime

# Phase data templates - defines structure for each phase
PHASE_DATA_TEMPLATES = {
    "data_import": {
        # File selection and validation
        "selected_files": {},  # {"train": "path/to/train.csv", "test": "path/to/test.csv"}
        "file_validation": {},  # {"train": {"valid": True, "rows": 1000}, "test": {"valid": True, "rows": 500}}
        "data_format": "",  # "csv", "mat", "excel"
        "processor_type": "",  # "STLA", "Arbin", "custom"
        
        # Progress tracking
        "import_progress": 0,  # 0-100
        "current_file": "",  # Currently processing file
        "files_processed": 0,
        "total_files": 0,
        "import_status": "pending",  # "pending", "processing", "completed", "error"
        "error_message": "",
        
        # Data info after processing
        "data_info": {
            "train_samples": 0,
            "test_samples": 0,
            "features": [],
            "target_columns": []
        },
        
        "ready_for_next": False
    },
    
    "data_augmentation": {
        # Augmentation configuration
        "augmentation_params": {
            "noise_level": 0.1,
            "time_shift": 0.0,
            "scaling_factor": 1.0,
            "rotation_angle": 0.0,
            "augmentation_ratio": 2.0  # How much to augment
        },
        
        # Preview data
        "preview_available": False,
        "preview_samples": [],  # List of sample augmented data for preview
        "original_samples": [],  # Original samples for comparison
        
        # Progress and status
        "augmentation_progress": 0,  # 0-100
        "augmentation_status": "pending",  # "pending", "processing", "completed", "error"
        "samples_augmented": 0,
        "total_samples": 0,
        "error_message": "",
        
        "ready_for_next": False
    },
    
    "hyperparameters": {
        # Model parameters
        "model_type": "LSTM",  # "LSTM", "GRU", "CNN", etc.
        "model_params": {
            # LSTM specific
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "bidirectional": False,
            
            # Training params
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "optimizer": "adam",
            "loss_function": "mse",
            
            # Validation
            "train_val_split": 0.8,
            "early_stopping_patience": 10,
            "early_stopping_delta": 0.001
        },
        
        # Device and resource preferences
        "device_preference": "auto",  # "cpu", "gpu", "auto"
        "use_mixed_precision": False,
        "num_workers": 4,
        
        # Validation results
        "validation_status": "pending",  # "pending", "valid", "invalid"
        "validation_errors": [],
        "parameter_conflicts": [],
        
        "ready_for_training": False
    },
    
    "training_setup": {
        # Task configuration
        "training_tasks": [],  # List of training task configurations
        "current_task_index": 0,
        "total_tasks": 0,
        
        # Setup status
        "setup_complete": False,
        "setup_errors": [],
        "resource_allocation": {
            "assigned_device": "",  # "cpu", "cuda:0", etc.
            "memory_allocated": 0,
            "estimated_duration": 0
        },
        
        "ready_for_training": False
    },
    
    "training": {
        # Current training state
        "training_active": False,
        "can_stop": False,
        "current_task": "",
        "task_start_time": "",
        "elapsed_time": 0,  # seconds
        
        # Epoch progress
        "current_epoch": 0,
        "total_epochs": 0,
        "epochs_completed": 0,
        
        # Current metrics
        "current_metrics": {
            "train_loss": 0.0,
            "val_loss": 0.0,
            "train_accuracy": 0.0,
            "val_accuracy": 0.0
        },
        
        # Best metrics
        "best_metrics": {
            "best_val_loss": float('inf'),
            "best_epoch": 0,
            "best_train_loss": float('inf'),
            "best_val_accuracy": 0.0
        },
        
        # Early stopping
        "early_stopping_patience": 10,
        "current_patience": 0,
        "patience_counter": 0,
        
        # Training history for plots (limited to recent epochs)
        "training_history": [],  # List of {"epoch": int, "train_loss": float, "val_loss": float, ...}
        "max_history_size": 1000,  # Keep last 1000 epochs
        
        # Logs (limited to recent entries)
        "training_logs": [],  # List of log messages
        "max_logs_size": 500,  # Keep last 500 log entries
        
        # Plot data (processed for visualization)
        "plots_data": {
            "loss_plot": {"epochs": [], "train_loss": [], "val_loss": []},
            "accuracy_plot": {"epochs": [], "train_acc": [], "val_acc": []},
            "learning_rate_plot": {"epochs": [], "lr": []}
        },
        
        # Training status
        "training_status": "pending",  # "pending", "running", "paused", "completed", "stopped", "error"
        "error_message": "",
        "completion_percentage": 0.0
    },
    
    "testing": {
        # Test configuration
        "test_data_path": "",
        "model_path": "",
        "test_batch_size": 32,
        
        # Test results
        "test_results": {
            "test_loss": 0.0,
            "test_accuracy": 0.0,
            "predictions": [],
            "ground_truth": [],
            "confusion_matrix": None
        },
        
        # Test progress
        "test_progress": 0,  # 0-100
        "test_status": "pending",  # "pending", "running", "completed", "error"
        "samples_processed": 0,
        "total_samples": 0,
        
        # Visualization data
        "plots_data": {
            "results_plot": {},
            "confusion_matrix_plot": {},
            "predictions_plot": {}
        },
        
        "ready_for_next": False
    }
}

def get_phase_template(phase: str) -> Dict[str, Any]:
    """
    Get a fresh copy of the phase template.
    
    Args:
        phase: Phase name (e.g., "data_import", "training", etc.)
        
    Returns:
        Fresh copy of the phase data template
    """
    if phase not in PHASE_DATA_TEMPLATES:
        raise ValueError(f"Unknown phase: {phase}. Available phases: {list(PHASE_DATA_TEMPLATES.keys())}")
    
    # Return a deep copy to avoid modifying the original template
    import copy
    return copy.deepcopy(PHASE_DATA_TEMPLATES[phase])

def get_all_phases() -> List[str]:
    """Get list of all available phases."""
    return list(PHASE_DATA_TEMPLATES.keys())

def initialize_phase_data(phase: str, **initial_values) -> Dict[str, Any]:
    """
    Initialize phase data with template and optional initial values.
    
    Args:
        phase: Phase name
        **initial_values: Initial values to override template defaults
        
    Returns:
        Initialized phase data
    """
    data = get_phase_template(phase)
    
    # Update with initial values if provided
    if initial_values:
        data.update(initial_values)
    
    # Add metadata
    data["_phase"] = phase
    data["_initialized_at"] = datetime.now().isoformat()
    data["_last_updated"] = data["_initialized_at"]
    
    return data

def validate_phase_data(phase: str, data: Dict[str, Any]) -> List[str]:
    """
    Validate phase data against template structure.
    
    Args:
        phase: Phase name
        data: Data to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    template = get_phase_template(phase)
    
    # Check for missing required keys
    for key in ["ready_for_next"]:  # Common required keys
        if key not in data:
            errors.append(f"Missing required key: {key}")
    
    # Phase-specific validations
    if phase == "training":
        if "current_epoch" in data and "total_epochs" in data:
            if data["current_epoch"] > data["total_epochs"]:
                errors.append("current_epoch cannot be greater than total_epochs")
    
    elif phase == "hyperparameters":
        if "model_params" in data:
            params = data["model_params"]
            if "learning_rate" in params and params["learning_rate"] <= 0:
                errors.append("learning_rate must be positive")
            if "batch_size" in params and params["batch_size"] <= 0:
                errors.append("batch_size must be positive")
    
    return errors
