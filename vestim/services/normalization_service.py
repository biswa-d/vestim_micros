import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
import os
import numpy as np

# Define a logger if you have a central logging setup, e.g.:
# from vestim.logger_config import setup_logger
# logger = setup_logger(__name__)
# For now, using print for simplicity, replace with logger.
# print("Normalization service module loaded.") # Commented out to reduce log clutter

def calculate_global_dataset_stats(data_items: list, feature_columns: list, data_reading_func=pd.read_csv, **read_kwargs):
    """
    Calculates global min and max statistics for specified features from a list of data sources.
    The data sources can be file paths or pre-loaded pandas DataFrames.

    Args:
        data_items (list): List of data sources. Each item can be a path to a data file (str)
                           or a pre-loaded pandas DataFrame.
        feature_columns (list): List of column names for which to calculate stats.
        data_reading_func (callable): Function to read a single data file if paths are provided
                                      (e.g., pd.read_csv, pd.read_excel).
                                      It must return a pandas DataFrame. Unused if data_items contains DataFrames.
        **read_kwargs: Additional keyword arguments to pass to data_reading_func. Unused if data_items contains DataFrames.

    Returns:
        dict: A dictionary with 'min' and 'max' keys, each containing a pandas Series
              with global min/max values for each feature_column.
              Returns None if data_items is empty or an error occurs.
    """
    if not data_items:
        print("Error: No data items (file paths or DataFrames) provided for stats calculation.")
        return None

    global_min = pd.Series(dtype=float)
    global_max = pd.Series(dtype=float)

    print(f"Calculating global stats for {len(data_items)} items and columns: {feature_columns}")

    for i, item in enumerate(data_items):
        df = None
        item_description = f"item {i+1}" # Default description
        try:
            if isinstance(item, pd.DataFrame):
                df = item
                item_description = f"DataFrame at index {i}"
                # print(f"Processing {item_description}")
            elif isinstance(item, str): # Assuming it's a file path
                item_description = f"file {item}"
                # print(f"Processing {item_description}")
                df = data_reading_func(item, **read_kwargs)
            else:
                print(f"Warning: Skipping item {i} in data_items as it's not a DataFrame or file path string.")
                continue

            if df is None or df.empty:
                print(f"Warning: DataFrame from {item_description} is None or empty. Skipping.")
                continue

            if not all(col in df.columns for col in feature_columns):
                print(f"Warning: Data from {item_description} is missing one or more feature columns: {feature_columns}. Skipping this item for those columns.")
                current_file_features = [col for col in feature_columns if col in df.columns]
                if not current_file_features:
                    continue
            else:
                current_file_features = feature_columns
            
            current_data = df[current_file_features].astype(float) # Ensure numeric types

            if global_min.empty:
                global_min = current_data.min()
                global_max = current_data.max()
            else:
                # Align series before comparison to handle missing columns in some files gracefully
                current_min_aligned, global_min_aligned = current_data.min().align(global_min, join='outer', fill_value=np.nan)
                current_max_aligned, global_max_aligned = current_data.max().align(global_max, join='outer', fill_value=np.nan)
                
                global_min = pd.concat([global_min_aligned, current_min_aligned], axis=1).min(axis=1, skipna=True)
                global_max = pd.concat([global_max_aligned, current_max_aligned], axis=1).max(axis=1, skipna=True)

        except Exception as e:
            print(f"Error processing data from {item_description} for stats: {e}")
            # Optionally, decide whether to continue or raise the error
            continue
    
    if global_min.empty or global_max.empty:
        print("Error: Could not calculate global stats. No valid data found or all files had errors.")
        return None

    print("Global stats calculation complete.")
    return {"min": global_min, "max": global_max}


def create_scaler_from_stats(global_stats, feature_columns, scaler_type='min_max'):
    """
    Creates and "fits" a scaler using pre-calculated global statistics.

    Args:
        global_stats (dict): Dictionary containing 'min' and 'max' pandas Series for features.
        feature_columns (list): The order of features for which the scaler is being created.
                                This ensures the scaler is fitted in the correct feature order.
        scaler_type (str): Type of scaler ('min_max' or 'z_score').

    Returns:
        A fitted scaler object (e.g., MinMaxScaler, StandardScaler) or None if error.
    """
    if not global_stats or 'min' not in global_stats or 'max' not in global_stats:
        print("Error: Invalid global_stats provided to create_scaler_from_stats.")
        return None
    
    # Ensure stats are Series and align them to feature_columns order
    try:
        stats_min = global_stats['min'].loc[feature_columns].values.reshape(1, -1)
        stats_max = global_stats['max'].loc[feature_columns].values.reshape(1, -1)
    except KeyError as e:
        print(f"Error: One or more feature_columns ({e}) not found in global_stats during scaler creation.")
        return None

    if scaler_type == 'min_max':
        scaler = MinMaxScaler()
        # Manually set the parameters of the scaler
        scaler.feature_range = (0, 1) # Default, can be parameterized
        
        # Calculate scale and min, handling division by zero for constant features
        scale = np.ones_like(stats_min, dtype=float)
        min_val = np.zeros_like(stats_min, dtype=float)
        
        diff = stats_max - stats_min
        
        # Where diff is not zero
        valid_scale_mask = diff != 0
        scale[valid_scale_mask] = 1.0 / diff[valid_scale_mask]
        min_val[valid_scale_mask] = -stats_min[valid_scale_mask] * scale[valid_scale_mask]
        
        # Where diff is zero (constant feature), map to 0
        # X_scaled = (X - X_min) / (X_max - X_min) -> if X_min == X_max, this should be 0
        # X_scaled = X * scale + min_val
        # If X = X_min, then X_min * scale_const + min_const = 0
        # Let scale_const = 1.0 (to avoid issues if X is 0 and scale is 0)
        # Then X_min * 1.0 + min_const = 0 => min_const = -X_min
        constant_feature_mask = diff == 0
        scale[constant_feature_mask] = 1.0
        min_val[constant_feature_mask] = -stats_min[constant_feature_mask]
        
        # Handle cases where min or max might be NaN (e.g., all-NaN column)
        # scale and min_val will become NaN automatically, which is fine.
        
        scaler.scale_ = scale
        scaler.min_ = min_val
        scaler.data_min_ = stats_min
        scaler.data_max_ = stats_max
        scaler.n_features_in_ = len(feature_columns)
        try: # Set feature_names_in_ if possible (sklearn >= 0.24 for MinMaxScaler)
            scaler.feature_names_in_ = np.array(feature_columns, dtype=object)
        except AttributeError:
            pass # Older sklearn, feature_names_in_ might not be settable or exist
    elif scaler_type == 'z_score':
        # For Z-score, we'd need mean and std. This function expects min/max.
        # If Z-score is needed, calculate_global_dataset_stats should also return mean and std.
        # And this function should be adapted.
        print(f"Error: Z-score scaling from min/max stats is not directly implemented. Mean/Std needed.")
        # scaler = StandardScaler()
        # scaler.mean_ = global_stats['mean'].loc[feature_columns].values
        # scaler.scale_ = global_stats['std'].loc[feature_columns].values
        # scaler.n_features_in_ = len(feature_columns)
        return None # Placeholder for Z-score
    else:
        print(f"Error: Unsupported scaler_type: {scaler_type}")
        return None
    
    print(f"Scaler ({scaler_type}) created and configured for features: {feature_columns}")
    return scaler

def save_scaler(scaler, directory, filename="scaler.joblib", job_id=None):
    """
    Saves a scaler object to a file using joblib and creates a JSON statistics file with global min/max values.

    Args:
        scaler: The scaler object to save.
        directory (str): The directory to save the scaler in.
        filename (str): The name of the file for the scaler.
        job_id (str, optional): The job ID for metadata context.

    Returns:
        str: Full path to the saved scaler file, or None if error.
    """
    if not scaler:
        print("Error: No scaler object provided to save.")
        return None
    try:
        import json
        
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
        
        # Save the scaler object
        scaler_path = os.path.join(directory, filename)
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")
        
        # Create JSON statistics file with global min/max values
        # Use the expected format: scaler_global_stats.txt
        base_name = filename.replace('.joblib', '')
        stats_filename = f"{base_name}_global_stats.txt"
        stats_path = os.path.join(directory, stats_filename)
        
        # Extract feature names and statistics
        feature_names = []
        global_min = {}
        global_max = {}
        
        if hasattr(scaler, 'feature_names_in_'):
            feature_names = list(scaler.feature_names_in_)
        elif hasattr(scaler, 'n_features_in_'):
            feature_names = [f"feature_{i}" for i in range(scaler.n_features_in_)]
        
        # Extract min/max values from scaler (these are the global stats used for fitting)
        if hasattr(scaler, 'data_min_') and hasattr(scaler, 'data_max_'):
            # MinMaxScaler - use data_min_ and data_max_ as global statistics
            for i, feature_name in enumerate(feature_names):
                global_min[feature_name] = float(scaler.data_min_[i])
                global_max[feature_name] = float(scaler.data_max_[i])
        elif hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
            # StandardScaler - estimate min/max using mean +/- 3*std as approximation
            for i, feature_name in enumerate(feature_names):
                mean_val = float(scaler.mean_[i])
                std_val = float(scaler.scale_[i])
                # Approximate range using 3-sigma rule
                global_min[feature_name] = mean_val - 3 * std_val
                global_max[feature_name] = mean_val + 3 * std_val
        
        # Determine job ID for comment
        if job_id is None:
            # Try to extract from directory path
            dir_parts = directory.replace('\\', '/').split('/')
            for part in dir_parts:
                if part.startswith('job_'):
                    job_id = part
                    break
            if job_id is None:
                job_id = "unknown"
        
        # Create JSON structure matching the requested format
        stats_data = {
            "comment": f"Global min/max statistics used for scaler '{filename}' on job '{job_id}'",
            "normalized_columns_for_stats": feature_names,
            "global_min": global_min,
            "global_max": global_max
        }
        
        # Save as JSON with proper formatting
        with open(stats_path, 'w') as f:
            json.dump(stats_data, f, indent=4)
        
        print(f"Scaler global statistics saved to {stats_path}")
        
        # Also create a human-readable text file for backward compatibility
        text_stats_filename = filename.replace('.joblib', '_statistics.txt')
        text_stats_path = os.path.join(directory, text_stats_filename)
        
        with open(text_stats_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("SCALER STATISTICS\n")
            f.write("=" * 60 + "\n")
            f.write(f"Scaler Type: {type(scaler).__name__}\n")
            f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Scaler File: {filename}\n")
            f.write(f"Job ID: {job_id}\n")
            f.write("-" * 60 + "\n\n")
            
            # Extract and display scaler statistics
            if hasattr(scaler, 'feature_names_in_'):
                f.write(f"Feature Count: {len(scaler.feature_names_in_)}\n")
                f.write(f"Feature Names: {list(scaler.feature_names_in_)}\n\n")
            elif hasattr(scaler, 'n_features_in_'):
                f.write(f"Feature Count: {scaler.n_features_in_}\n\n")
            
            # MinMaxScaler statistics
            if hasattr(scaler, 'data_min_') and hasattr(scaler, 'data_max_'):
                f.write("COLUMN STATISTICS (MinMaxScaler)\n")
                f.write("-" * 40 + "\n")
                for i, feature_name in enumerate(feature_names):
                    min_val = scaler.data_min_[i]
                    max_val = scaler.data_max_[i]
                    data_range = max_val - min_val
                    f.write(f"{feature_name}:\n")
                    f.write(f"  Min: {min_val:.6f}\n")
                    f.write(f"  Max: {max_val:.6f}\n")
                    f.write(f"  Range: {data_range:.6f}\n")
                    f.write(f"  Scale Factor: {scaler.scale_[i]:.6f}\n\n")
            
            # StandardScaler statistics
            elif hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
                f.write("COLUMN STATISTICS (StandardScaler)\n")
                f.write("-" * 40 + "\n")
                for i, feature_name in enumerate(feature_names):
                    mean_val = scaler.mean_[i]
                    scale_val = scaler.scale_[i]
                    f.write(f"{feature_name}:\n")
                    f.write(f"  Mean: {mean_val:.6f}\n")
                    f.write(f"  Std Dev: {scale_val:.6f}\n")
                    f.write(f"  Scale Factor: {1/scale_val:.6f}\n\n")
            
            f.write("-" * 60 + "\n")
            f.write("USAGE NOTES:\n")
            f.write("- This scaler transforms data to range [0, 1] for MinMaxScaler\n")
            f.write("- This scaler standardizes data to mean=0, std=1 for StandardScaler\n")
            f.write("- Use the same scaler for inference to maintain consistency\n")
            f.write("- Load scaler with: joblib.load('{}')".format(filename))
        
        print(f"Scaler text statistics saved to {text_stats_path}")
        return scaler_path
        
    except Exception as e:
        print(f"Error saving scaler: {e}")
        return None

def load_scaler(scaler_path):
    """
    Loads a scaler object from a file using joblib.

    Args:
        scaler_path (str): Path to the scaler file.

    Returns:
        The loaded scaler object, or None if error.
    """
    try:
        if not os.path.exists(scaler_path):
            print(f"Error: Scaler file not found at {scaler_path}")
            return None
        scaler = joblib.load(scaler_path)
        print(f"Scaler loaded from {scaler_path}")
        return scaler
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return None

def transform_data(data_df, scaler, feature_columns, all_numeric_columns=None):
    """
    Transforms specified columns of a DataFrame using a pre-fitted scaler.
    If all_numeric_columns is provided, it will normalize all of them. Otherwise, it defaults
    to transforming only the feature_columns.

    Args:
        data_df (pd.DataFrame): The DataFrame to transform.
        scaler: The pre-fitted scaler object.
        feature_columns (list): List of column names that were used to fit the scaler.
        all_numeric_columns (list, optional): List of all numeric columns to apply the transformation to.
                                             If None, defaults to feature_columns.

    Returns:
        pd.DataFrame: DataFrame with specified columns transformed, or original if error.
    """
    if not scaler:
        print("Error: No scaler provided for transform_data.")
        return data_df

    # Determine which columns to transform
    if all_numeric_columns:
        cols_to_transform = [col for col in all_numeric_columns if col in data_df.columns]
    else:
        cols_to_transform = [col for col in feature_columns if col in data_df.columns]

    if not cols_to_transform:
        print("Warning: No columns to transform found in the DataFrame.")
        return data_df

    # Filter out columns that are not in the scaler's feature list
    # This prevents errors when trying to normalize excluded columns
    if hasattr(scaler, 'feature_names_in_'):
        scaler_features = list(scaler.feature_names_in_)
        cols_to_transform = [col for col in cols_to_transform if col in scaler_features]
        
        # Log any excluded columns for debugging
        excluded_cols = [col for col in (all_numeric_columns or feature_columns) 
                        if col in data_df.columns and col not in scaler_features]
        # if excluded_cols:
        #     print(f"Info: Skipping columns not in scaler features: {excluded_cols}")
    
    if not cols_to_transform:
        print("Warning: No valid columns to transform found after filtering.")
        return data_df

    try:
        data_copy = data_df.copy()
        
        # Get the scaler features in the correct order
        scaler_features = list(scaler.feature_names_in_)
        
        # Only transform columns that are both in the scaler and should be transformed
        # Filter cols_to_transform to only include columns that are in scaler_features
        actual_cols_to_transform = [col for col in cols_to_transform if col in scaler_features]
        
        if not actual_cols_to_transform:
            print("Warning: No columns to transform after filtering by scaler features.")
            return data_copy
        
        # Create a DataFrame with ALL scaler features in the correct order for transformation
        # This ensures the scaler gets the data in the expected format
        scaler_data = data_copy[scaler_features].astype(float)
        
        # Transform all scaler features (scaler expects all features it was trained on)
        transformed_data = scaler.transform(scaler_data)
        
        # Create a DataFrame with the transformed data
        transformed_df = pd.DataFrame(transformed_data, index=data_copy.index, columns=scaler_features)
        
        # Update the original DataFrame only with the columns that were meant to be transformed
        # This preserves non-normalized columns like timestamps
        for col in actual_cols_to_transform:
            data_copy[col] = transformed_df[col]
        
        # The following logs are too verbose for production, but useful for debugging.
        # print(f"Data transformed for columns: {actual_cols_to_transform}")
        # non_transformed_cols = [col for col in data_copy.columns if col not in actual_cols_to_transform]
        # print(f"Preserved non-normalized columns: {non_transformed_cols}")
        
        # # Debug: Check if timestamp columns are preserved
        # timestamp_cols = [col for col in non_transformed_cols if 'time' in col.lower().replace(" ", "")]
        # if timestamp_cols:
        #     print(f"Timestamp columns preserved: {timestamp_cols}")
        # else:
        #     print("No timestamp columns found in preserved columns")
            
        return data_copy
    except Exception as e:
        print(f"Error during data transformation: {e}")
        return data_df # Return original on error

def inverse_transform_data(data_df, scaler, feature_columns):
    """
    Inverse transforms specified columns of a DataFrame using a pre-fitted scaler.

    Args:
        data_df (pd.DataFrame): The DataFrame with transformed data.
        scaler: The pre-fitted scaler object used for original transformation.
        feature_columns (list): List of column names to inverse_transform.

    Returns:
        pd.DataFrame: DataFrame with specified columns inverse_transformed, or original if error.
    """
    if not scaler:
        print("Error: No scaler provided for inverse_transform_data.")
        return data_df
    if not all(col in data_df.columns for col in feature_columns):
        missing_cols = [col for col in feature_columns if col not in data_df.columns]
        print(f"Warning: inverse_transform_data - DataFrame is missing columns: {missing_cols}. Skipping inverse transformation for these.")
        transformable_cols = [col for col in feature_columns if col in data_df.columns]
        if not transformable_cols:
            return data_df
    else:
        transformable_cols = feature_columns
        
    try:
        data_copy = data_df.copy()
        # Convert to NumPy array, ensuring correct column order for inverse_transform
        data_to_inverse_transform_df = data_copy[transformable_cols]
        
        # Inverse transform the data
        inverse_transformed_data = scaler.inverse_transform(data_to_inverse_transform_df)
        
        # Create a new DataFrame with the inverse-transformed data
        inverse_transformed_df = pd.DataFrame(inverse_transformed_data, index=data_copy.index, columns=transformable_cols)
        
        # Update the original DataFrame with the inverse-transformed columns
        for col in transformable_cols:
            data_copy[col] = inverse_transformed_df[col]
            
        print(f"Data inverse_transformed for columns: {transformable_cols}")
        return data_copy
    except Exception as e:
        print(f"Error during data inverse transformation: {e}")
        return data_df # Return original on error

def inverse_transform_single_column(normalized_values, scaler, target_column, normalized_columns):
    """
    Safely inverse transform a single column using scaler parameters directly.
    This avoids the need to create full DataFrames with zeros for other columns.
    
    Args:
        normalized_values: numpy array or tensor of normalized values for the target column
        scaler: The fitted scaler object
        target_column: Name of the target column to denormalize
        normalized_columns: List of all columns that were normalized (used to find target column index)
    
    Returns:
        numpy array of denormalized values, or original values if error occurs
    """
    try:
        import numpy as np
        
        # Convert to numpy if it's a tensor
        if hasattr(normalized_values, 'cpu'):
            values = normalized_values.cpu().numpy() if normalized_values.is_cuda else normalized_values.numpy()
        else:
            values = np.array(normalized_values)
        
        values = values.flatten()
        
        # Find the index of the target column in the scaler
        if hasattr(scaler, 'feature_names_in_'):
            if target_column not in scaler.feature_names_in_:
                print(f"Warning: {target_column} not found in scaler features. Returning original values.")
                return values
            target_col_index = list(scaler.feature_names_in_).index(target_column)
        else:
            # Fallback to using normalized_columns list
            if target_column not in normalized_columns:
                print(f"Warning: {target_column} not found in normalized columns. Returning original values.")
                return values
            target_col_index = normalized_columns.index(target_column)
        
        # Apply inverse transformation based on scaler type
        if hasattr(scaler, 'data_min_') and hasattr(scaler, 'data_max_'):
            # MinMaxScaler
            # Handle both 1D and 2D scaler parameter arrays
            data_min_array = np.array(scaler.data_min_).flatten()
            data_max_array = np.array(scaler.data_max_).flatten()
            data_min = data_min_array[target_col_index]
            data_max = data_max_array[target_col_index]
            denormalized = values * (data_max - data_min) + data_min
            
            # Debug: Check for unusual normalized values that might indicate scaling issues
            if len(values) > 0:
                val_min, val_max = np.min(values), np.max(values)
                if val_min < -0.1 or val_max > 1.1:
                    print(f"WARNING: Normalized values outside expected [0,1] range: min={val_min:.6f}, max={val_max:.6f}")
                    print(f"This suggests the model output is not properly normalized or there's a scaling issue")
            
            # Only print range info once per column per session to reduce log spam
            if not hasattr(inverse_transform_single_column, '_printed_ranges'):
                inverse_transform_single_column._printed_ranges = set()
            
            range_key = f"{target_column}_{data_min:.6f}_{data_max:.6f}"
            if range_key not in inverse_transform_single_column._printed_ranges:
                print(f"MinMaxScaler inverse transform for {target_column}: range=({data_min:.6f}, {data_max:.6f}), scale_factor={(data_max - data_min):.6f}")
                inverse_transform_single_column._printed_ranges.add(range_key)
                
        elif hasattr(scaler, 'scale_') and hasattr(scaler, 'mean_'):
            # StandardScaler - CORRECTED FORMULA
            # Handle both 1D and 2D scaler parameter arrays
            scale_array = np.array(scaler.scale_).flatten()
            mean_array = np.array(scaler.mean_).flatten()
            scale = scale_array[target_col_index]  # This is actually the std deviation
            mean = mean_array[target_col_index]
            denormalized = values * scale + mean  # For StandardScaler: X_original = X_normalized * std + mean
            print(f"StandardScaler inverse transform: scale(std)={scale:.6f}, mean={mean:.6f}")
        else:
            print(f"Warning: Unknown scaler type {type(scaler)}. Returning original values.")
            denormalized = values
        
        return denormalized
        
    except Exception as e:
        print(f"Error in inverse_transform_single_column: {e}")
        return normalized_values.flatten() if hasattr(normalized_values, 'flatten') else normalized_values

def denormalize_predictions_and_targets(predictions, targets, scaler, target_column, normalized_columns, return_as='numpy'):
    """
    Denormalizes predictions and target values for testing/evaluation phase.
    Handles both individual predictions and batch predictions safely.
    
    Args:
        predictions: numpy array, tensor, or list of prediction values (normalized)
        targets: numpy array, tensor, or list of target values (normalized) 
        scaler: The fitted scaler object used during training
        target_column: Name of the target column that was normalized
        normalized_columns: List of all columns that were normalized (for index lookup)
        return_as: 'numpy', 'list', or 'tensor' - format to return results
    
    Returns:
        tuple: (denormalized_predictions, denormalized_targets) in requested format
    """
    try:
        import numpy as np
        
        # Convert inputs to numpy arrays
        if hasattr(predictions, 'cpu'):
            pred_np = predictions.cpu().numpy() if predictions.is_cuda else predictions.numpy()
        else:
            pred_np = np.array(predictions)
        
        if hasattr(targets, 'cpu'):
            targ_np = targets.cpu().numpy() if targets.is_cuda else targets.numpy()
        else:
            targ_np = np.array(targets)
        
        # Flatten to ensure 1D arrays
        pred_np = pred_np.flatten()
        targ_np = targ_np.flatten()
        
        # Denormalize both using the single column function
        denorm_pred = inverse_transform_single_column(pred_np, scaler, target_column, normalized_columns)
        denorm_targ = inverse_transform_single_column(targ_np, scaler, target_column, normalized_columns)
        
        # Return in requested format
        if return_as == 'list':
            return denorm_pred.tolist(), denorm_targ.tolist()
        elif return_as == 'tensor':
            try:
                import torch
                return torch.tensor(denorm_pred), torch.tensor(denorm_targ)
            except ImportError:
                print("Warning: PyTorch not available, returning as numpy arrays")
                return denorm_pred, denorm_targ
        else:  # numpy (default)
            return denorm_pred, denorm_targ
            
    except Exception as e:
        print(f"Error in denormalize_predictions_and_targets: {e}")
        # Return original values in case of error
        return predictions, targets

def save_denormalized_results_with_metadata(predictions, targets, timestamps, metadata_df, 
                                          scaler, target_column, normalized_columns, 
                                          output_file_path, target_units=""):
    """
    Saves denormalized predictions and targets along with timestamps and metadata to a CSV file.
    This is useful for the testing phase where you want to save results with original timestamps.
    
    Args:
        predictions: Normalized prediction values
        targets: Normalized target values  
        timestamps: Timestamp column from original data
        metadata_df: DataFrame containing other metadata columns (IDs, status, etc.)
        scaler: The fitted scaler object
        target_column: Name of the target column
        normalized_columns: List of normalized column names
        output_file_path: Path where to save the results CSV
        target_units: String describing the units (e.g., "mV", "% SOC") for column naming
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import pandas as pd
        
        # Denormalize the predictions and targets
        denorm_pred, denorm_targ = denormalize_predictions_and_targets(
            predictions, targets, scaler, target_column, normalized_columns, return_as='numpy'
        )
        
        # Create results DataFrame
        results_df = pd.DataFrame()
        
        # Add timestamps if provided
        if timestamps is not None:
            if hasattr(timestamps, 'values'):
                results_df['Timestamp'] = timestamps.values
            else:
                results_df['Timestamp'] = timestamps
        
        # Add metadata columns if provided
        if metadata_df is not None and not metadata_df.empty:
            # Reset index to ensure alignment
            metadata_reset = metadata_df.reset_index(drop=True)
            for col in metadata_reset.columns:
                if col.lower() not in ['timestamp', 'time']:  # Avoid duplicate timestamp
                    results_df[col] = metadata_reset[col].values
        
        # Add denormalized predictions and targets
        pred_col_name = f'Predicted_{target_column}'
        true_col_name = f'True_{target_column}'
        
        if target_units:
            pred_col_name += f' ({target_units})'
            true_col_name += f' ({target_units})'
        
        results_df[pred_col_name] = denorm_pred
        results_df[true_col_name] = denorm_targ
        
        # Calculate error metrics
        rmse = np.sqrt(np.mean((denorm_pred - denorm_targ)**2))
        mae = np.mean(np.abs(denorm_pred - denorm_targ))
        mape = np.mean(np.abs((denorm_pred - denorm_targ) / denorm_targ)) * 100
        
        results_df[f'Absolute_Error ({target_units})'] = np.abs(denorm_pred - denorm_targ)
        results_df[f'Relative_Error (%)'] = np.abs((denorm_pred - denorm_targ) / denorm_targ) * 100
        
        # Save to CSV
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        results_df.to_csv(output_file_path, index=False)
        
        print(f"Results saved to: {output_file_path}")
        print(f"Denormalized RMSE: {rmse:.4f} {target_units}")
        print(f"Denormalized MAE: {mae:.4f} {target_units}")
        print(f"Denormalized MAPE: {mape:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"Error saving denormalized results: {e}")
        return False

if __name__ == '__main__':
    print("Normalization Service - Production Module")
    print("This module provides normalization and denormalization functions for the ML pipeline.")
    print("Import this module to use its functions in your application.")