from flask import Flask, request, jsonify
import os
import scipy.io
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route('/convert', methods=['POST'])
def convert_files():
    data = request.get_json()
    files = data['files']  # List of file paths
    train_output_dir = data['train_output_dir']
    test_output_dir = data['test_output_dir']

    # Debugging: Print received file paths
    print(f"Received file paths for conversion: {files}")

    for file_path in files:
        # Determine the correct output folder
        if 'train' in file_path:
            output_folder = train_output_dir
        elif 'test' in file_path:
            output_folder = test_output_dir
        else:
            output_folder = train_output_dir  # Default to train if not specified
        
        # Debugging: Check if the file exists before converting
        if os.path.exists(file_path):
            print(f"Converting file: {file_path}")
            # Process file based on extension
            if file_path.endswith('.mat'):
                convert_mat_to_csv(file_path, output_folder)
            elif file_path.endswith('.csv'):
                convert_csv_to_csv(file_path, output_folder)
            elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                convert_excel_to_csv(file_path, output_folder)
            else:
                print(f"Skipping file {file_path}: Unsupported file format")
        else:
            print(f"File does not exist: {file_path}")

    return jsonify({"message": "Files have been converted"}), 200

def convert_mat_to_csv(mat_file, output_folder):
    """Convert MATLAB .mat file to CSV format with expected columns."""
    data = scipy.io.loadmat(mat_file)
    if 'meas' in data:
        meas = data['meas'][0, 0]
        Timestamp = meas['Time'].flatten()
        Voltage = meas['Voltage'].flatten()
        Current = meas['Current'].flatten()
        Temp = meas['Battery_Temp_degC'].flatten()
        SOC = meas['SOC'].flatten()

        combined_data = np.column_stack((Timestamp, Voltage, Current, Temp, SOC))
        header = ['Timestamp', 'Voltage', 'Current', 'Temp', 'SOC']
        
        csv_file_name = os.path.join(output_folder, os.path.splitext(os.path.basename(mat_file))[0] + '.csv')
        np.savetxt(csv_file_name, combined_data, delimiter=",", header=",".join(header), comments='', fmt='%s')
        print(f'Data successfully written to {csv_file_name}')
    else:
        print(f'Skipping file {mat_file}: "meas" field not found')

def convert_csv_to_csv(csv_file, output_folder):
    """
    Read CSV file and ensure it has the expected format with the required columns.
    If necessary, it will transform the data to match the expected format.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Check if the dataframe has the required columns or mappable columns
        required_columns = ['Timestamp', 'Voltage', 'Current', 'Temp', 'SOC']
        
        # Map columns if they exist with different names (common in different data sources)
        column_mapping = {
            'Time': 'Timestamp',
            'Potential': 'Voltage', 
            'Voltage(V)': 'Voltage',
            'Current(A)': 'Current',
            'Temperature': 'Temp', 
            'Temperature(C)': 'Temp',
            'State of Charge': 'SOC'
        }
        
        # Rename columns if they exist with different names
        df = df.rename(columns=column_mapping)
        
        # Check if all required columns are present
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Warning: CSV file {csv_file} is missing required columns: {missing_columns}")
            # You might want to add logic to handle missing columns
        
        # Save to the output folder
        output_file = os.path.join(output_folder, os.path.basename(csv_file))
        df.to_csv(output_file, index=False)
        print(f'Data successfully written to {output_file}')
        
    except Exception as e:
        print(f"Error processing CSV file {csv_file}: {str(e)}")

def convert_excel_to_csv(excel_file, output_folder):
    """
    Convert Excel file to CSV format with expected columns.
    Handles both .xlsx and .xls formats.
    """
    try:
        # Read the Excel file
        df = pd.read_excel(excel_file)
        
        # Same column mapping logic as in convert_csv_to_csv
        required_columns = ['Timestamp', 'Voltage', 'Current', 'Temp', 'SOC']
        
        column_mapping = {
            'Time': 'Timestamp',
            'Potential': 'Voltage', 
            'Voltage(V)': 'Voltage',
            'Current(A)': 'Current',
            'Temperature': 'Temp', 
            'Temperature(C)': 'Temp',
            'State of Charge': 'SOC'
        }
        
        # Rename columns if they exist with different names
        df = df.rename(columns=column_mapping)
        
        # Check if all required columns are present
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Warning: Excel file {excel_file} is missing required columns: {missing_columns}")
            # You might want to add logic to handle missing columns
        
        # Save to the output folder with .csv extension
        base_name = os.path.splitext(os.path.basename(excel_file))[0]
        output_file = os.path.join(output_folder, base_name + '.csv')
        df.to_csv(output_file, index=False)
        print(f'Data successfully written to {output_file}')
        
    except Exception as e:
        print(f"Error processing Excel file {excel_file}: {str(e)}")

if __name__ == '__main__':
    app.run(port=5002, debug=True)