import tkinter as tk
from tkinter import filedialog, Label, Toplevel, messagebox
import os
import json
from src.gui.src.training_setup_gui import VEstimTrainSetupGUI
from src.gateway.src.job_manager import JobManager
from src.gateway.src.hyper_param_manager import VEstimHyperParamManager

# Initialize the JobManager
job_manager = JobManager()

class VEstimHyperParamGUI:
    def __init__(self, master):
        self.master = master
        self.params = {}  # Initialize an empty params dictionary
        self.job_manager = job_manager  # Use the shared JobManager instance
        self.hyper_param_manager = VEstimHyperParamManager()  # Initialize HyperParamManager
        self.param_entries = {}  # To store the entry widgets for parameters

        self.setup_window()
        self.build_gui()

    def setup_window(self):
        self.master.title("VEstim")
        self.master.geometry("700x600")
        resources_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources')
        icon_path = os.path.join(resources_path, 'icon.ico')
        if os.path.exists(icon_path):
            self.master.iconbitmap(icon_path)
        else:
            print("Icon file not found. Make sure 'icon.ico' is in the correct directory.")

    def build_gui(self):
        for widget in self.master.winfo_children():
            widget.destroy()

        title_label = tk.Label(self.master, text="Select Hyperparameters for LSTM Model", font=("Helvetica", 14, "bold"))
        title_label.grid(row=0, column=0, columnspan=4, pady=20)

        tk.Button(self.master, text="Open Hyperparameter Guide", command=self.open_guide).grid(row=1, column=0, columnspan=4, pady=10)

        self.add_param_widgets()

        start_button = tk.Button(self.master, text="Create Training Tasks", command=self.proceed_to_training, bg="blue", fg="white")
        start_button.grid(row=11, column=1, columnspan=2, pady=15, sticky="ew")

        load_button = tk.Button(self.master, text="Load Params from File", command=self.load_params_from_json)
        load_button.grid(row=9, column=1, columnspan=2, pady=5, sticky="ew")

        # save_button = tk.Button(self.master, text="Save Params to File", command=self.save_params_to_json)
        # save_button.grid(row=10, column=1, columnspan=2, pady=5, sticky="ew")

        self.master.grid_columnconfigure(1, weight=1)
        self.master.grid_columnconfigure(2, weight=1)

    def add_param_widgets(self):
        hyperparameters = [
            {"label": "Layers:", "default": self.params.get("LAYERS", "1"), "tooltip": "Number of LSTM layers in the model", "param": "LAYERS"},
            {"label": "Hidden Units:", "default": self.params.get("HIDDEN_UNITS", "10"), "tooltip": "Number of units in each LSTM layer", "param": "HIDDEN_UNITS"},
            {"label": "Mini-batches:", "default": self.params.get("BATCH_SIZE", "100"), "tooltip": "Number of mini-batches to use during training", "param": "BATCH_SIZE"},
            {"label": "Max Epochs:", "default": self.params.get("MAX_EPOCHS", "5000"), "tooltip": "Maximum number of epochs to train the model", "param": "MAX_EPOCHS"},
            {"label": "Initial LR:", "default": self.params.get("INITIAL_LR", "0.00001"), "tooltip": "The starting learning rate for the optimizer", "param": "INITIAL_LR"},
            {"label": "LR Drop Factor:", "default": self.params.get("LR_DROP_FACTOR", "0.5"), "tooltip": "Factor by which the learning rate is reduced after Drop Period", "param": "LR_DROP_FACTOR"},
            {"label": "LR Drop Period:", "default": self.params.get("LR_DROP_PERIOD", "10"), "tooltip": "The number of epochs after which the learning rate drops", "param": "LR_DROP_PERIOD"},
            {"label": "Validation Patience:", "default": self.params.get("VALID_PATIENCE", "10"), "tooltip": "Number of epochs to wait for validation improvement before early stopping", "param": "VALID_PATIENCE"},
            {"label": "Validation Freq:", "default": self.params.get("ValidFrequency", "3"), "tooltip": "How often (in epochs) to perform validation", "param": "ValidFrequency"},
            {"label": "Lookback:", "default": self.params.get("LOOKBACK", "400"), "tooltip": "Number of previous time steps to consider for each timestep", "param": "LOOKBACK"},
            {"label": "Repetitions:", "default": self.params.get("REPETITIONS", "1"), "tooltip": "Number of times to repeat the entire training process with randomized initial parameters", "param": "REPETITIONS"},
        ]

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

            self.param_entries[param_name] = entry

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
        # Collect updated parameters from the user inputs
        new_params = {param: entry.get() for param, entry in self.param_entries.items()}
        self.update_params(new_params)
        self.hyper_param_manager.save_params()  # Save the params using the manager

        for widget in self.master.winfo_children():
            widget.destroy()

        self.master.title("VEstim - Training LSTM Model")

        # Transition to the training GUI
        VEstimTrainSetupGUI(self.master, new_params, job_manager)

    def load_params_from_json(self):
        filepath = filedialog.askopenfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if filepath:
            try:
                # Load and validate parameters using the manager
                self.params = self.hyper_param_manager.load_params(filepath)
                self.update_gui_with_loaded_params()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load parameters: {str(e)}")

    def update_params(self, new_params):
        try:
            self.hyper_param_manager.update_params(new_params)
            # self.params = self.hyper_param_manager.get_current_params()
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameter input: {str(e)}")
        self.update_gui_with_loaded_params()

    def update_gui_with_loaded_params(self):
        for param_name, entry in self.param_entries.items():
            if param_name in self.params:
                value = ', '.join(map(str, self.params[param_name])) if isinstance(self.params[param_name], list) else str(self.params[param_name])
                entry.delete(0, tk.END)
                entry.insert(0, value)

    def save_params_to_json(self):
        new_params = {param: entry.get() for param, entry in self.param_entries.items()}
        self.update_params(new_params)
        self.hyper_param_manager.save_params()  # Save the params using the manager

    # def update_params(self, new_params):
    #     self.params.update(new_params)
    #     self.hyper_param_manager.update_params(new_params)  # Update the params in the manager

    # def update_gui_with_loaded_params(self):
    #     for param_name, entry in self.param_entries.items():
    #         if param_name in self.params:
    #             # If the parameter is a list, join it into a string for display
    #             value = ', '.join(map(str, self.params[param_name])) if isinstance(self.params[param_name], list) else self.params[param_name]
    #             entry.delete(0, tk.END)
    #             entry.insert(0, value)


    def open_guide(self, event=None):
        resources_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources')
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
    gui = VEstimHyperParamGUI(root)
    root.mainloop()
