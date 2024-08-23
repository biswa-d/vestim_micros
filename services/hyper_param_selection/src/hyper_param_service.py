from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

class VEstimHyperParamService:
    def __init__(self, params=None):
        self.params = params if params else {}

    def load_hyperparams(self):
        # This would load hyperparameters from the backend storage (e.g., a file, database)
        # For now, it simulates a load with a simple example
        try:
            response = requests.get('http://127.0.0.1:5003/load_hyperparams')
            if response.status_code == 200:
                self.params = response.json().get('hyperparams', {})
                print(f"Loaded hyperparameters: {self.params}")
            else:
                print("Failed to load hyperparameters")
        except requests.exceptions.RequestException as e:
            print(f"Error loading hyperparameters: {e}")

    def save_hyperparams(self):
        # This would save hyperparameters to the backend storage (e.g., a file, database)
        # For now, it simulates a save with a simple example
        try:
            response = requests.post('http://127.0.0.1:5003/save_hyperparams', json=self.params)
            if response.status_code == 200:
                print("Hyperparameters saved successfully")
            else:
                print("Failed to save hyperparameters")
        except requests.exceptions.RequestException as e:
            print(f"Error saving hyperparameters: {e}")

    def update_params(self, new_params):
        self.params.update(new_params)

# Flask route to save hyperparameters
@app.route('/save_hyperparams', methods=['POST'])
def save_hyperparams():
    hyperparams = request.get_json()
    # Here you would implement the logic to save the hyperparams to a database or file
    return jsonify({"message": "Hyperparameters saved successfully"}), 200

# Flask route to load hyperparameters
@app.route('/load_hyperparams', methods=['GET'])
def load_hyperparams():
    # Here you would implement the logic to load the hyperparams from a database or file
    return jsonify({"hyperparams": {"LAYERS": 3, "HIDDEN_UNITS": 100}}), 200

if __name__ == '__main__':
    app.run(port=5003, debug=True)
