from flask import Flask, request, jsonify
import os
import scipy.io
import numpy as np

app = Flask(__name__)

@app.route('/convert', methods=['POST'])
def convert_files():
    data = request.get_json()
    files = data['files']  # List of .mat file paths
    train_output_dir = data['train_output_dir']
    test_output_dir = data['test_output_dir']

    # Debugging: Print received file paths
    print(f"Received file paths for conversion: {files}")

    for file_path in files:
        if file_path.endswith('.mat'):
            # Determine the correct output folder
            if 'train' in file_path:
                output_folder = train_output_dir
            elif 'test' in file_path:
                output_folder = test_output_dir
            
             # Debugging: Check if the file exists before converting
            if os.path.exists(file_path):
                print(f"Converting file: {file_path}")
                convert_mat_to_csv(file_path, output_folder)
            else:
                print(f"File does not exist: {file_path}")

    return jsonify({"message": "Files have been converted"}), 200

def convert_mat_to_csv(mat_file, output_folder):
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
        np.savetxt(csv_file_name, combined_data, delimiter=",", header=",".join(header), comments='', fmt='%s')
    else:
        print(f'Skipping file {mat_file}: "meas" field not found')

if __name__ == '__main__':
    app.run(port=5002, debug=True)