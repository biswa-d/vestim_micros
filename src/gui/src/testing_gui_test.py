import tkinter as tk
from tkinter import ttk
from threading import Thread
import time
from queue import Queue, Empty
from src.gateway.src.testing_manager_test import VEstimTestingManager
from src.gateway.src.job_manager import JobManager
from src.gateway.src.training_setup_manager_test import VEstimTrainingSetupManager
from src.gateway.src.hyper_param_manager import VEstimHyperParamManager

class VEstimTestingGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("VEstim Tool - Model Testing")
        self.master.geometry("900x700")

        self.job_manager = JobManager()
        self.testing_manager = VEstimTestingManager()
        self.hyper_param_manager = VEstimHyperParamManager()
        self.training_setup_manager = VEstimTrainingSetupManager()
        
        # Add the param_labels dictionary for user-friendly display
        self.param_labels = {
            "LAYERS": "Layers",
            "HIDDEN_UNITS": "Hidden Units",
            "BATCH_SIZE": "Batch Size",
            "MAX_EPOCHS": "Max Epochs",
            "INITIAL_LR": "Initial Learning Rate",
            "LR_DROP_FACTOR": "LR Drop Factor",
            "LR_DROP_PERIOD": "LR Drop Period",
            "VALID_PATIENCE": "Validation Patience",
            "ValidFrequency": "Validation Frequency",
            "LOOKBACK": "Lookback Sequence Length",
            "REPETITIONS": "Repetitions"
        }
        
        self.queue = Queue()  # Queue to handle test results
        self.task_list = self.training_setup_manager.get_task_list()

        self.timer_running = False
        self.start_time = None

        self.setup_ui()

    def get_hyper_params_text(self):
        """Returns a formatted string of hyperparameters."""
        params = self.hyper_param_manager.get_hyper_params()
        # Use self.param_labels to convert internal parameter names to user-friendly names
        params_text = ", ".join([f"{self.param_labels.get(k, k)}: {v}" for k, v in params.items()])
        return f"Hyperparameters: {params_text}"
    

    def build_gui(self):
        # Clear existing content
        for widget in self.master.winfo_children():
            widget.destroy()

        # Set window title and size
        self.master.title("VEstim - Testing LSTM Models")
        self.master.geometry("900x600")
        self.master.minsize(900, 600)

        # Create main frame
        main_frame = tk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        main_frame.pack_propagate(False)

        # Title Label
        title_label = tk.Label(main_frame, text="Testing LSTM Models", font=("Helvetica", 14, "bold"), fg="#006400")
        title_label.pack(pady=5)

        # Hyperparameters Frame
        self.hyperparam_frame = tk.Frame(main_frame)
        self.hyperparam_frame.pack(fill=tk.X, pady=5)

        # Display hyperparameters
        params = self.hyper_param_manager.get_current_params()  # Get current hyperparameters
        self.display_hyperparameters(params)

        # Status Label
        self.status_label = tk.Label(main_frame, text="Preparing test data...", font=("Helvetica", 12))
        self.status_label.pack(pady=10)

        # Testing Timer
        self.time_label = tk.Label(main_frame, text="Testing Time: 00h:00m:00s", font=("Helvetica", 10))
        self.time_label.pack(pady=5)

        # Results Frame
        self.results_frame = tk.Frame(main_frame)
        self.results_frame.pack(fill=tk.BOTH, expand=True)

        # Add a scrollbar
        self.scrollbar = tk.Scrollbar(self.results_frame)
        self.scrollbar.pack(side="right", fill="y")

        # Results Table
        self.tree = ttk.Treeview(self.results_frame, columns=("Sl.No", "Model", "RMS Error (mV)", "MAE (mV)", "MAPE (%)", "R²", "Plot"), show="headings", yscrollcommand=self.scrollbar.set)
        self.tree.pack(fill="both", expand=True)

        # Configure the scrollbar
        self.scrollbar.config(command=self.tree.yview)

        # Set headings
        for col in self.tree["columns"]:
            self.tree.heading(col, text=col)

        # Start Button
        self.start_button = tk.Button(main_frame, text="Start Testing", command=self.start_testing)
        self.start_button.pack(pady=10)

        # Open Results Folder Button (hidden initially)
        self.open_results_button = tk.Button(main_frame, text="Open Results Folder", command=self.open_results_folder)
        self.open_results_button.pack(pady=10)
        self.open_results_button.pack_forget()  # Hide initially


    def get_hyper_params_text(self):
        """Returns a formatted string of hyperparameters."""
        params = self.hyper_param_manager.get_current_params()
        params_text = ", ".join([f"{self.param_labels.get(k, k)}: {v}" for k, v in params.items()])
        return f"Hyperparameters: {params_text}"
    
    def display_hyperparameters(self, params):
        # Clear previous widgets in the hyperparam_frame
        for widget in self.hyperparam_frame.winfo_children():
            widget.destroy()

        # Get the parameter items
        param_items = [(self.param_labels.get(param, param), value) for param, value in params.items()]

        # Split the parameters into five columns
        columns = [param_items[i::5] for i in range(5)]  # Splits into five columns

        # Display each column with labels
        for col_num, column in enumerate(columns):
            col_frame = tk.Frame(self.hyperparam_frame)
            col_frame.grid(row=0, column=col_num, padx=5)
            for row_num, (param, value) in enumerate(column):
                param_label = tk.Label(col_frame, text=f"{param}: ", font=("Helvetica", 10))  # Regular font for label
                value_label = tk.Label(col_frame, text=f"{value}", font=("Helvetica", 10, "bold"))  # Bold font for value

                # Use grid to ensure both labels stay on the same line
                param_label.grid(row=row_num, column=0, sticky='w')
                value_label.grid(row=row_num, column=1, sticky='w')

        # Centering the hyperparameters table
        self.hyperparam_frame.grid_columnconfigure(0, weight=1)
        self.hyperparam_frame.grid_columnconfigure(len(columns) - 1, weight=1)


    def start_testing(self):
        self.start_button.config(state=tk.DISABLED)
        self.status_label.config(text="Preparing test data...")
        self.start_time = time.time()

        # Start the testing process in a separate thread
        test_thread = Thread(target=self.run_testing)
        test_thread.start()

    def update_timer(self):
        if self.timer_running:
            elapsed_time = time.time() - self.start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.time_label.config(text=f"Testing Time: {int(hours):02}h:{int(minutes):02}m:{int(seconds):02}s")
            self.master.after(1000, self.update_timer)

    def run_testing(self):
        # Update status to indicate test data preparation
        self.update_status("Creating test tasks...")

        # Start the timer
        self.timer_running = True
        self.update_timer()

        # Run the testing process and populate the queue with results
        self.testing_manager.start_testing()

        # Process the testing tasks sequentially and display results
        while True:
            try:
                result = self.queue.get_nowait()
                self.add_result_row(result)
            except Empty:
                time.sleep(1)
                if not self.testing_manager.is_testing_running():
                    break

        # Update status and stop the timer when all tests are complete
        self.update_status("All tests completed!")
        self.timer_running = False

        # Show the "Open Results Folder" button after testing completes
        self.open_results_button.pack()

    def update_status(self, message):
        self.status_label.config(text=message)

    def add_result_row(self, result):
        # Extract necessary info from the result dictionary
        model_name = os.path.basename(result["model_path"])
        rms_error = result["rms_error_mv"]
        mae = result["mae_mv"]
        mape = result["mape"]
        r2 = result["r2"]

        # Create a plot button for each row
        plot_button = tk.Button(self.tree, text="Plot", command=lambda m=model_name: self.plot_result(m))

        # Insert the result row into the treeview
        self.tree.insert("", "end", values=(len(self.tree.get_children()), model_name, rms_error, mae, mape, r2))

    def plot_result(self, model_name):
        # Implement logic to plot the results for the selected model
        pass

    def open_results_folder(self):
        # Open the results folder using the default file explorer
        results_folder = self.job_manager.get_test_results_folder()
        if os.path.exists(results_folder):
            os.startfile(results_folder)

if __name__ == "__main__":
    root = tk.Tk()
    gui = TestingGUI(root)
    root.mainloop()
