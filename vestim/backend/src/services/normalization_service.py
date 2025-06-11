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

def save_scaler(scaler, directory, filename="scaler.joblib"):
    """
    Saves a scaler object to a file using joblib.

    Args:
        scaler: The scaler object to save.
        directory (str): The directory to save the scaler in.
        filename (str): The name of the file for the scaler.

    Returns:
        str: Full path to the saved scaler file, or None if error.
    """
    if not scaler:
        print("Error: No scaler object provided to save.")
        return None
    try:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
        
        scaler_path = os.path.join(directory, filename)
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")
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

def transform_data(data_df, scaler, feature_columns):
    """
    Transforms specified columns of a DataFrame using a pre-fitted scaler.

    Args:
        data_df (pd.DataFrame): The DataFrame to transform.
        scaler: The pre-fitted scaler object.
        feature_columns (list): List of column names to transform.

    Returns:
        pd.DataFrame: DataFrame with specified columns transformed, or original if error.
    """
    if not scaler:
        print("Error: No scaler provided for transform_data.")
        return data_df
    if not all(col in data_df.columns for col in feature_columns):
        missing_cols = [col for col in feature_columns if col not in data_df.columns]
        print(f"Warning: transform_data - DataFrame is missing columns: {missing_cols}. Skipping transformation for these.")
        # Transform only available columns
        transformable_cols = [col for col in feature_columns if col in data_df.columns]
        if not transformable_cols:
            return data_df # No columns to transform
    else:
        transformable_cols = feature_columns

    try:
        data_copy = data_df.copy()
        # Ensure data is float before transforming
        data_to_transform_df = data_copy[transformable_cols].astype(float)
        # Convert to NumPy array to avoid UserWarning about feature names
        # The order of columns in transformable_cols is derived from feature_columns,
        # which is the order used to set up the scaler.
        data_to_transform_np = data_to_transform_df.to_numpy()
        
        transformed_np = scaler.transform(data_to_transform_np)
        
        # Assign back to the DataFrame
        data_copy[transformable_cols] = transformed_np
        print(f"Data transformed for columns: {transformable_cols}")
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
        data_to_inverse_transform_np = data_to_inverse_transform_df.to_numpy()

        inverse_transformed_np = scaler.inverse_transform(data_to_inverse_transform_np)
        
        data_copy[transformable_cols] = inverse_transformed_np
        print(f"Data inverse_transformed for columns: {transformable_cols}")
        return data_copy
    except Exception as e:
        print(f"Error during data inverse transformation: {e}")
        return data_df # Return original on error

if __name__ == '__main__':
    # Example Usage (Illustrative - replace with actual file paths and columns)
    print("\n--- Example Usage of Normalization Service ---")
    
    # Create dummy CSV files for testing
    dummy_dir = "dummy_data_for_norm_test"
    os.makedirs(dummy_dir, exist_ok=True)
    
    data1 = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50], 'C': [100, 200, 150, 250, 180]})
    data1_path = os.path.join(dummy_dir, "data1.csv")
    data1.to_csv(data1_path, index=False)
    
    data2 = pd.DataFrame({'A': [0, 6, 2], 'B': [5, 55, 25], 'D': [1,2,3]}) # 'C' is missing, 'D' is extra
    data2_path = os.path.join(dummy_dir, "data2.csv")
    data2.to_csv(data2_path, index=False)

    data3 = pd.DataFrame({'A': [2.5, 3.5], 'B': [22, 33], 'C': [120, 220]})
    data3_path = os.path.join(dummy_dir, "data3.csv")
    data3.to_csv(data3_path, index=False)

    all_files = [data1_path, data2_path, data3_path]
    features_to_normalize = ['A', 'B', 'C'] # Note: 'C' is missing in data2

    # 1. Calculate global stats
    print("\n1. Calculating global stats...")
    global_stats = calculate_global_dataset_stats(all_files, features_to_normalize)
    
    if global_stats:
        print("Global Min:\n", global_stats['min'])
        print("Global Max:\n", global_stats['max'])

        # 2. Create scaler from stats
        print("\n2. Creating scaler...")
        # Important: Pass feature_columns in the desired, consistent order
        scaler = create_scaler_from_stats(global_stats, features_to_normalize, scaler_type='min_max')

        if scaler:
            # 3. Save the scaler
            print("\n3. Saving scaler...")
            scaler_save_dir = os.path.join(dummy_dir, "scalers")
            saved_scaler_path = save_scaler(scaler, scaler_save_dir, filename="my_global_scaler.joblib")

            if saved_scaler_path:
                # 4. Load the scaler (as if in a new process/testing phase)
                print("\n4. Loading scaler...")
                loaded_scaler = load_scaler(saved_scaler_path)

                if loaded_scaler:
                    # 5. Transform new data (or training data in Pass 2)
                    print("\n5. Transforming data (using data1 as example)...")
                    original_df_to_transform = pd.read_csv(data1_path)
                    print("Original Data1:\n", original_df_to_transform)
                    
                    transformed_df = transform_data(original_df_to_transform.copy(), loaded_scaler, features_to_normalize)
                    print("Transformed Data1:\n", transformed_df)

                    # 6. Inverse transform data
                    print("\n6. Inverse transforming data...")
                    inverse_transformed_df = inverse_transform_data(transformed_df.copy(), loaded_scaler, features_to_normalize)
                    print("Inverse Transformed Data1 (should be close to original):\n", inverse_transformed_df)

    # Clean up dummy files
    # import shutil
    # shutil.rmtree(dummy_dir)
    # print(f"\nCleaned up dummy directory: {dummy_dir}")
    print("\n--- End of Example Usage ---")