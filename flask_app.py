from flask import Flask, request, jsonify
import os
import time

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_files():
    train_folder = request.files.getlist('train_folder')
    test_folder = request.files.getlist('test_folder')
    
    job_id = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(os.getcwd(), 'output')
    os.makedirs(output_dir, exist_ok=True)
    job_folder = os.path.join(output_dir, f"Job_{job_id}")
    os.makedirs(job_folder, exist_ok=True)
    
    # Process files (simplified)
    for file in train_folder:
        file.save(os.path.join(job_folder, file.filename))
    for file in test_folder:
        file.save(os.path.join(job_folder, file.filename))
    
    return jsonify({"message": "Files have been organized", "job_folder": job_folder}), 200

if __name__ == '__main__':
    app.run(port=5001)