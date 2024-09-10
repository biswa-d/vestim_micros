#----------------------------------------------------------------------------------------
# Description: Contains the code for the hyperparamter selection on a GUI screen which
# will be used to train the LSTM model for estimation of Voltage for a Lithium-ion battery. This script is to test the standalone functionality of the GUI.
#
# Created On: Wed Aug 07 2024 16:01:19
# Author: Biswanath Dehury
# Company: Dr. Phil Kollmeyer's Battery Lab at McMaster University
# Copyright (c) 2024 Biswanath Dehury, Dr. Phil Kollmeyer's Battery Lab at McMaster University
#----------------------------------------------------------------------------------------


import tkinter as tk
from tkinter import Toplevel, Label
from src.gui.training_gui import VEstimTrainingGUI  # Use the refactored class for training GUI
import webbrowser
import pkg_resources

class VEstimHyperParamGUI:
    def __init__(self, master, params):
        self.master = master
        self.params = params
        self.param_entries = {}  # Dictionary to hold the entry widgets
        self.build_gui()

    # Utility function to add tooltips
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
    # Define a list of hyperparameters with their default values and tooltips
        hyperparameters = [
            {"label": "Layers:", "default": "1", "tooltip": "Number of LSTM layers in the model", "param": "LAYERS"},
            {"label": "Hidden Units:", "default": "10", "tooltip": "Number of units in each LSTM layer", "param": "HIDDEN_UNITS"},
            {"label": "Mini-batches:", "default": "100", "tooltip": "Number of mini-batches to use during training", "param": "BATCH_SIZE"},
            {"label": "Max Epochs:", "default": "5000", "tooltip": "Maximum number of epochs to train the model", "param": "MAX_EPOCHS"},
            {"label": "Initial LR:", "default": "0.00001", "tooltip": "The starting learning rate for the optimizer", "param": "INITIAL_LR"},
            {"label": "LR Drop Factor:", "default": "0.5", "tooltip": "Factor by which the learning rate is reduced", "param": "LR_DROP_FACTOR"},
            {"label": "LR Drop Period:", "default": "10", "tooltip": "The number of epochs after which the learning rate drops", "param": "LR_DROP_PERIOD"},
            {"label": "Validation Patience:", "default": "10", "tooltip": "Number of epochs to wait for validation improvement before early stopping", "param": "VALID_PATIENCE"},
            {"label": "Validation Freq:", "default": "3", "tooltip": "How often (in epochs) to perform validation", "param": "ValidFrequency"},
            {"label": "Lookback:", "default": "400", "tooltip": "Number of previous time steps to consider for each sequence", "param": "LOOKBACK"},
            {"label": "Repetitions:", "default": "1", "tooltip": "Number of times to repeat the entire training process", "param": "REPETITIONS"},
        ]

        # Dynamically create labels and entry widgets
        for idx, param in enumerate(hyperparameters):
            row = 2 + idx // 2  # Positioning rows
            column = (idx % 2) * 2  # Positioning columns

            label_text = param["label"]
            default_value = param["default"]
            tooltip_text = param["tooltip"]
            param_name = param["param"]  # Get the param key

            label = tk.Label(self.master, text=label_text)
            label.grid(row=row, column=column, sticky='w', padx=10, pady=8)
            self.create_tooltip(label, tooltip_text)

            entry = tk.Entry(self.master)
            entry.insert(0, default_value)
            entry.grid(row=row, column=column + 1, padx=8, pady=10, sticky='ew')

            # Store entry widgets in a dictionary using the param name as the key
            self.param_entries[param_name] = entry

    def proceed_to_training(self):
        # Update the params dictionary with new hyperparameters
        for param_name, entry in self.param_entries.items():
            value = entry.get()
            if param_name in ["INITIAL_LR", "LR_DROP_FACTOR", "LR_DROP_PERIOD"]:  # Add other parameters that should be floats
                self.params[param_name] = float(value)
            else:
                self.params[param_name] = int(value)

        print("Selected parameters:", self.params)

        # Clear the existing content from the master window
        for widget in self.master.winfo_children():
            widget.destroy()

        # Reset window title
        self.master.title("VEstim - Training LSTM Model")

        # Start the training process in the existing window
        VEstimTrainingGUI(self.master, self.params)
    
    # Function to open the hyperparameter guide
    def open_guide(self, event=None):
        # Opens a PDF or a document guide
        pdf_path = pkg_resources.resource_filename(__name__, '../resources/hyper_param_guide.pdf')
        webbrowser.open('file://' + pdf_path)
