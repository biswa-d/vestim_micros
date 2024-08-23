import tkinter as tk
from tkinter import filedialog, Label, Toplevel
import os
import requests
from services.hyper_param_selection.src.hyper_param_service import VEstimHyperParamService

class VEstimHyperParamGUI:
    def __init__(self, master, params):
        self.master = master
        self.service = VEstimHyperParamService(params)
        self.service.load_hyperparams()
        self.setup_window()
        self.build_gui()

    def setup_window(self):
        self.master.title("VEstim")
        self.master.geometry("700x600")  # Increased window size
        resources_path = os.path.join(os.path.dirname(__file__), 'resources')
        icon_path = os.path.join(resources_path, 'icon.ico')
        if os.path.exists(icon_path):
            self.master.iconbitmap(icon_path)
        else:
            print("Icon file not found. Make sure 'icon.ico' is in the correct directory.")

    # def build_gui(self):
    #     for widget in self.master.winfo_children():
    #         widget.destroy()

    #     title_label = tk.Label(self.master, text="Select Hyperparameters for LSTM Model", font=("Helvetica", 12, "bold"))
    #     title_label.grid(row=0, column=0, columnspan=4, pady=20)

    #     tk.Button(self.master, text="Open Hyperparameter Guide", command=self.open_guide).grid(row=1, column=0, columnspan=4, pady=10)

    #     self.add_param_widgets()

    #     start_button = tk.Button(self.master, text="Start Training", command=self.proceed_to_training, bg="blue", fg="white")
    #     start_button.grid(row=8, column=1, columnspan=2, pady=15, sticky="ew")

    #     load_button = tk.Button(self.master, text="Load Params from JSON", command=self.load_params_from_json)
    #     load_button.grid(row=9, column=1, columnspan=2, pady=5, sticky="ew")

    #     save_button = tk.Button(self.master, text="Save Params to JSON", command=self.save_params_to_json)
    #     save_button.grid(row=10, column=1, columnspan=2, pady=5, sticky="ew")

    #     self.master.grid_columnconfigure(1, weight=1)
    #     self.master.grid_columnconfigure(2, weight=1)
    def build_gui(self):
        for widget in self.master.winfo_children():
            widget.destroy()

        title_label = tk.Label(self.master, text="Select Hyperparameters for LSTM Model", font=("Helvetica", 14, "bold"))
        title_label.grid(row=0, column=0, columnspan=4, pady=20)

        tk.Button(self.master, text="Open Hyperparameter Guide", command=self.open_guide).grid(row=1, column=0, columnspan=4, pady=10)

        self.add_param_widgets()

        start_button = tk.Button(self.master, text="Start Training", command=self.proceed_to_training, bg="blue", fg="white")
        start_button.grid(row=11, column=1, columnspan=2, pady=15, sticky="ew")

        load_button = tk.Button(self.master, text="Load Params from JSON", command=self.load_params_from_json)
        load_button.grid(row=9, column=1, columnspan=2, pady=5, sticky="ew")

        save_button = tk.Button(self.master, text="Save Params to JSON", command=self.save_params_to_json)
        save_button.grid(row=10, column=1, columnspan=2, pady=5, sticky="ew")

        self.master.grid_columnconfigure(1, weight=1)
        self.master.grid_columnconfigure(2, weight=1)


    def add_param_widgets(self):
        hyperparameters = [
            {"label": "Layers:", "default": self.service.params.get("LAYERS", "1"), "tooltip": "Number of LSTM layers in the model", "param": "LAYERS"},
            {"label": "Hidden Units:", "default": self.service.params.get("HIDDEN_UNITS", "10"), "tooltip": "Number of units in each LSTM layer", "param": "HIDDEN_UNITS"},
            {"label": "Mini-batches:", "default": self.service.params.get("BATCH_SIZE", "100"), "tooltip": "Number of mini-batches to use during training", "param": "BATCH_SIZE"},
            {"label": "Max Epochs:", "default": self.service.params.get("MAX_EPOCHS", "5000"), "tooltip": "Maximum number of epochs to train the model", "param": "MAX_EPOCHS"},
            {"label": "Initial LR:", "default": self.service.params.get("INITIAL_LR", "0.00001"), "tooltip": "The starting learning rate for the optimizer", "param": "INITIAL_LR"},
            {"label": "LR Drop Factor:", "default": self.service.params.get("LR_DROP_FACTOR", "0.5"), "tooltip": "Factor by which the learning rate is reduced after Drop Period", "param": "LR_DROP_FACTOR"},
            {"label": "LR Drop Period:", "default": self.service.params.get("LR_DROP_PERIOD", "10"), "tooltip": "The number of epochs after which the learning rate drops", "param": "LR_DROP_PERIOD"},
            {"label": "Validation Patience:", "default": self.service.params.get("VALID_PATIENCE", "10"), "tooltip": "Number of epochs to wait for validation improvement before early stopping", "param": "VALID_PATIENCE"},
            {"label": "Validation Freq:", "default": self.service.params.get("ValidFrequency", "3"), "tooltip": "How often (in epochs) to perform validation", "param": "ValidFrequency"},
            {"label": "Lookback:", "default": self.service.params.get("LOOKBACK", "400"), "tooltip": "Number of previous time steps to consider for each timestep", "param": "LOOKBACK"},
            {"label": "Repetitions:", "default": self.service.params.get("REPETITIONS", "1"), "tooltip": "Number of times to repeat the entire training process with randomised initial parameters", "param": "REPETITIONS"},
        ]

        self.master.grid_columnconfigure(0, weight=1)  # First column
        self.master.grid_columnconfigure(1, weight=1)  # Second column
        self.master.grid_columnconfigure(2, weight=1)  # Third column
        self.master.grid_columnconfigure(3, weight=1)  # Fourth column

        for idx, param in enumerate(hyperparameters):
            row = 2 + idx // 2
            column = (idx % 2) * 2

            label_text = param["label"]
            default_value = param["default"]
            tooltip_text = param["tooltip"]
            param_name = param["param"]

            label = tk.Label(self.master, text=label_text)
            label.grid(row=row, column=column, sticky='w', padx=10, pady=15)
            self.create_tooltip(label, tooltip_text)

            entry = tk.Entry(self.master)
            entry.insert(0, default_value)
            entry.grid(row=row, column=column + 1, padx=20, pady=15, sticky='ew')

            self.service.param_entries[param_name] = entry

    def create_tooltip(self, widget, text):
        tooltip = None

        def on_enter(event):
            nonlocal tooltip
            tooltip = Toplevel(widget)
            tooltip.overrideredirect(True)
            tooltip.geometry(f"+{event.x_root + 10}+{event.y_root + 10}")
            Label(tooltip, text=text, background="yellow", relief='solid', borderwidth=1).pack()

        def on_leave(event):
            nonlocal tooltip
            if tooltip:
                tooltip.destroy()
            tooltip = None

        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    def proceed_to_training(self):
        new_params = {param: entry.get() for param, entry in self.service.param_entries.items()}
        self.service.update_params(new_params)
        self.service.save_hyperparams()

        for widget in self.master.winfo_children():
            widget.destroy()

        self.master.title("VEstim - Training LSTM Model")

        # VEstimTrainingGUI(self.master, self.service.params)

    import requests  # Ensure that requests is imported for HTTP communication

    def load_params_from_json(self):
        filepath = filedialog.askopenfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if filepath:
            # Make a POST request to load params from the JSON file via the Flask service
            response = requests.post('http://127.0.0.1:5003/load_params_from_json', json={'filepath': filepath})
            if response.status_code == 200:
                self.service.params = response.json().get('params', {})
                self.update_gui_with_loaded_params()
            else:
                print(f"Failed to load parameters: {response.text}")

    def save_params_to_json(self):
        # Default save operation (saves to the output/job_id/hyperparams.json)
        job_id = "job_id"  # Replace with actual job ID generation logic
        output_dir = "output"  # This is the default directory where the job folders are created

        new_params = {param: entry.get() for param, entry in self.service.param_entries.items()}
        self.service.update_params(new_params)

        # Make a POST request to save params to the default location
        response = requests.post('http://127.0.0.1:5003/save_hyperparams', json={
            'params': self.service.params,
            'output_dir': output_dir,
            'job_id': job_id
        })
        if response.status_code == 200:
            print(response.json().get('message', 'Parameters saved successfully'))
        else:
            print(f"Failed to save parameters: {response.text}")

    def save_params_to_custom_location(self):
        filepath = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if filepath:
            new_params = {param: entry.get() for param, entry in self.service.param_entries.items()}
            self.service.update_params(new_params)

            # Make a POST request to save params to the specified file location
            response = requests.post('http://127.0.0.1:5003/save_params_to_file', json={
                'params': self.service.params,
                'filepath': filepath
            })
            if response.status_code == 200:
                print(response.json().get('message', 'Parameters saved successfully'))
            else:
                print(f"Failed to save parameters: {response.text}")

    def update_gui_with_loaded_params(self):
        for param_name, entry in self.service.param_entries.items():
            if param_name in self.service.params:
                entry.delete(0, tk.END)
                entry.insert(0, self.service.params[param_name])


    def open_guide(self, event=None):
        resources_path = os.path.join(os.path.dirname(__file__), 'resources')
        pdf_path = os.path.join(resources_path, 'hyper_param_guide.pdf')
        if os.path.exists(pdf_path):
            try:
                os.startfile(pdf_path)
            except Exception as e:
                print(f"Failed to open PDF: {e}")
        else:
            print("PDF guide not found. Make sure 'hyper_param_guide.pdf' is in the correct directory.")

if __name__ == "__main__":
    root = tk.Tk()
    params = {}
    gui = VEstimHyperParamGUI(root, params)
    root.mainloop()
