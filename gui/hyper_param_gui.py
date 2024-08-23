import tkinter as tk
from tkinter import Toplevel, Label
import requests  # For making HTTP requests to Flask service
import webbrowser
import os
from services.hyper_param_selection.src.hyper_param_service import VEstimHyperParamService  # Import the service class

class VEstimHyperParamGUI:
    def __init__(self, master, params):
        self.master = master
        self.service = VEstimHyperParamService(params)  # Initialize the service class
        self.load_hyperparams()  # Load hyperparameters from the service
        self.build_gui()

    # Function to handle hyperparameter selection
    def build_gui(self):
        # Clear the existing content from the master window
        for widget in self.master.winfo_children():
            widget.destroy()

        # Title
        title_label = tk.Label(self.master, text="Select Hyperparameters for LSTM Model", font=("Helvetica", 12, "bold"))
        title_label.grid(row=0, column=0, columnspan=4, pady=20)

        # Button to open detailed guide at the top
        tk.Button(self.master, text="Open Hyperparameter Guide", command=self.open_guide).grid(row=1, column=0, columnspan=4, pady=10)

        # Define hyperparameter labels and entry widgets
        self.add_param_widgets()

        # Button to start training
        start_button = tk.Button(self.master, text="Start Training", command=self.proceed_to_training, bg="blue", fg="white")
        start_button.grid(row=8, column=1, columnspan=2, pady=15, sticky="ew")

        # Adjust the button width by setting the grid column's weight to control the expansion
        self.master.grid_columnconfigure(1, weight=1)
        self.master.grid_columnconfigure(2, weight=1)

    # Function to add parameter widgets
    def add_param_widgets(self):
        hyperparameters = [
            {"label": "Layers:", "default": self.service.params.get("LAYERS", "1"), "tooltip": "Number of LSTM layers in the model", "param": "LAYERS"},
            {"label": "Hidden Units:", "default": self.service.params.get("HIDDEN_UNITS", "10"), "tooltip": "Number of units in each LSTM layer", "param": "HIDDEN_UNITS"},
            {"label": "Mini-batches:", "default": self.service.params.get("BATCH_SIZE", "100"), "tooltip": "Number of mini-batches to use during training", "param": "BATCH_SIZE"},
            {"label": "Max Epochs:", "default": self.service.params.get("MAX_EPOCHS", "5000"), "tooltip": "Maximum number of epochs to train the model", "param": "MAX_EPOCHS"},
            {"label": "Initial LR:", "default": self.service.params.get("INITIAL_LR", "0.00001"), "tooltip": "The starting learning rate for the optimizer", "param": "INITIAL_LR"},
            {"label": "LR Drop Factor:", "default": self.service.params.get("LR_DROP_FACTOR", "0.5"), "tooltip": "Factor by which the learning rate is reduced", "param": "LR_DROP_FACTOR"},
            {"label": "LR Drop Period:", "default": self.service.params.get("LR_DROP_PERIOD", "10"), "tooltip": "The number of epochs after which the learning rate drops", "param": "LR_DROP_PERIOD"},
            {"label": "Validation Patience:", "default": self.service.params.get("VALID_PATIENCE", "10"), "tooltip": "Number of epochs to wait for validation improvement before early stopping", "param": "VALID_PATIENCE"},
            {"label": "Validation Freq:", "default": self.service.params.get("ValidFrequency", "3"), "tooltip": "How often (in epochs) to perform validation", "param": "ValidFrequency"},
            {"label": "Lookback:", "default": self.service.params.get("LOOKBACK", "400"), "tooltip": "Number of previous time steps to consider for each sequence", "param": "LOOKBACK"},
            {"label": "Repetitions:", "default": self.service.params.get("REPETITIONS", "1"), "tooltip": "Number of times to repeat the entire training process", "param": "REPETITIONS"},
        ]

        for idx, param in enumerate(hyperparameters):
            row = 2 + idx // 2
            column = (idx % 2) * 2

            label_text = param["label"]
            default_value = param["default"]
            tooltip_text = param["tooltip"]
            param_name = param["param"]

            label = tk.Label(self.master, text=label_text)
            label.grid(row=row, column=column, sticky='w', padx=10, pady=8)
            self.service.create_tooltip(label, tooltip_text)  # Use the service function

            entry = tk.Entry(self.master)
            entry.insert(0, default_value)
            entry.grid(row=row, column=column + 1, padx=8, pady=10, sticky='ew')

            # Store entry widgets in the service class dictionary
            self.service.param_entries[param_name] = entry

    def load_hyperparams(self):
        try:
            response = requests.get('http://127.0.0.1:5003/load_hyperparams')
            if response.status_code == 200:
                self.service.params = response.json().get('hyperparams', {})
                print(f"Loaded hyperparameters: {self.service.params}")
            else:
                print("Failed to load hyperparameters")
        except requests.exceptions.RequestException as e:
            print(f"Error loading hyperparameters: {e}")

    def proceed_to_training(self):
        # Update the params dictionary using the service function
        self.service.update_params()

        # Clear the existing content from the master window
        for widget in self.master.winfo_children():
            widget.destroy()

        # Reset window title
        self.master.title("VEstim - Training LSTM Model")

        # Start the training process in the existing window
        # VEstimTrainingGUI(self.master, self.service.params)

    # Function to open the hyperparameter guide
    def open_guide(self, event=None):
        # Opens a PDF or a document guide
        pdf_path = os.path.join(os.path.dirname(__file__), '../resources/hyper_param_guide.pdf')
        webbrowser.open('file://' + os.path.abspath(pdf_path))

if __name__ == "__main__":
    root = tk.Tk()
    params = {}  # Initialize with default params or load from a file
    gui = VEstimHyperParamGUI(root, params)
    root.mainloop()
