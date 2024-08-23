import os
import shutil
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_files():
    data = request.get_json()
    train_files = data['train_files']
    test_files = data['test_files']
    job_id = data['job_id']

    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(root_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    job_folder = os.path.join(output_dir, f"Job_{job_id}")
    os.makedirs(job_folder, exist_ok=True)

    # Create subdirectories for raw and processed data
    train_raw_folder = os.path.join(job_folder, 'train', 'raw_data')
    train_processed_folder = os.path.join(job_folder, 'train', 'processed_data')
    test_raw_folder = os.path.join(job_folder, 'test', 'raw_data')
    test_processed_folder = os.path.join(job_folder, 'test', 'processed_data')

    os.makedirs(train_raw_folder, exist_ok=True)
    os.makedirs(train_processed_folder, exist_ok=True)
    os.makedirs(test_raw_folder, exist_ok=True)
    os.makedirs(test_processed_folder, exist_ok=True)

    # Copy training files to the raw_data folder
    for file_path in train_files:
        dest_path = os.path.join(train_raw_folder, os.path.basename(file_path))
        shutil.copy(file_path, dest_path)
        print(f'Copied {file_path} to {dest_path}') # Debugging

    # Copy testing files to the raw_data folder
    for file_path in test_files:
        dest_path = os.path.join(test_raw_folder, os.path.basename(file_path))
        shutil.copy(file_path, dest_path)
        print(f'Copied {file_path} to {dest_path}') # Debugging

    # Prepare the file paths for conversion
    train_file_paths = [os.path.join(train_raw_folder, os.path.basename(file)) for file in train_files]
    test_file_paths = [os.path.join(test_raw_folder, os.path.basename(file)) for file in test_files]

    # Debugging: Print file paths before sending
    print(f"Train file paths: {train_file_paths}")
    print(f"Test file paths: {test_file_paths}")

    # Combine paths and send to data conversion service
    all_file_paths_for_conversion = train_file_paths + test_file_paths

    response = requests.post(
        "http://127.0.0.1:5002/convert",
        json={'files': all_file_paths_for_conversion, 'train_output_dir': train_processed_folder, 'test_output_dir': test_processed_folder}
    )

    if response.status_code == 200:
        return jsonify({"message": "Files have been uploaded and converted", "job_folder": job_folder}), 200
    else:
        return jsonify({"message": "Failed to convert files"}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)