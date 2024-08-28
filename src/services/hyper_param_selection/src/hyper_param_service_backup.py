from flask import Flask, request, jsonify
import json
import os

app = Flask(__name__)

class VEstimHyperParamService:
    def __init__(self, params=None):
        self.params = params if params else {}
        self.param_entries = {}

    def load_params_from_json(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'r') as file:
                self.params = json.load(file)
            return self.params
        else:
            return {}

    def save_hyperparams(self, new_params, output_dir, job_id):
        self.params.update(new_params)
        os.makedirs(output_dir, exist_ok=True)
        param_file = os.path.join(output_dir, job_id, 'hyperparams.json')
        os.makedirs(os.path.dirname(param_file), exist_ok=True)
        with open(param_file, 'w') as file:
            json.dump(self.params, file, indent=4)
        return param_file

    def save_params_to_file(self, new_params, filepath):
        self.params.update(new_params)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as file:
            json.dump(self.params, file, indent=4)
        return filepath
    
    # Define the update_params method to update the hyperparameters
    def update_params(self, new_params):
        self.params.update(new_params)
    
param_service = VEstimHyperParamService()

@app.route('/load_params_from_json', methods=['POST'])
def load_params_from_json():
    data = request.get_json()
    filepath = data.get('filepath')
    params = param_service.load_params_from_json(filepath)
    if params:
        return jsonify({"params": params}), 200
    else:
        return jsonify({"message": "File not found"}), 404

@app.route('/save_hyperparams', methods=['POST'])
def save_hyperparams():
    data = request.get_json()
    new_params = data.get('params')
    output_dir = data.get('output_dir')
    job_id = data.get('job_id')

    param_file = param_service.save_hyperparams(new_params, output_dir, job_id)
    return jsonify({"message": f"Parameters saved to {param_file}"}), 200

@app.route('/save_params_to_file', methods=['POST'])
def save_params_to_file():
    data = request.get_json()
    new_params = data.get('params')
    filepath = data.get('filepath')

    param_file = param_service.save_params_to_file(new_params, filepath)
    return jsonify({"message": f"Parameters saved to {param_file}"}), 200

if __name__ == '__main__':
    app.run(port=5003, debug=True)
