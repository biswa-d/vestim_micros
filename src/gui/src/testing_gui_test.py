import tkinter as tk
from tkinter import ttk
from threading import Thread
import time, os
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

        # Dictionary for user-friendly display of parameter names
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
        self.timer_running = False
        self.start_time = None

        # Immediately build the GUI and start testing
        self.build_gui()
        self.start_testing()

    def build_gui(self):
        # Clear existing content
        for widget in self.master.winfo_children():
            widget.destroy()

        # Set window title and size
        self.master.title("VEstim - Testing LSTM Models")
        self.master.geometry("900x700")
        self.master.minsize(900, 700)

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
        params = self.hyper_param_manager.get_current_params()
        self.display_hyperparameters(params)

        # Testing Timer
        self.time_label = tk.Label(main_frame, text="Testing Time: 00h:00m:00s", font=("Helvetica", 10), fg="blue")
        self.time_label.pack(pady=5)

        # Testing Result Summary Label
        result_summary_label = tk.Label(main_frame, text="Testing Result Summary", font=("Helvetica", 12, "bold"))
        result_summary_label.pack(pady=5)

        # Results Frame
        self.results_frame = tk.Frame(main_frame)
        self.results_frame.pack(fill=tk.BOTH, expand=True)

        # Add a scrollbar
        self.scrollbar = tk.Scrollbar(self.results_frame)
        self.scrollbar.pack(side="right", fill="y")

       # Set up the style for bold headers with an underline
        style = ttk.Style()
        style.configure("Treeview.Heading", font=("Helvetica", 10, "bold"), relief="solid", borderwidth=2)
        # Add an underline effect
        style.map("Treeview.Heading", background=[("!pressed", "white")], relief=[("!pressed", "groove")])
        # Set up the style for the Treeview rows
        style.configure("Treeview", font=("Helvetica", 10))  # Set the font for the rows

        # Results Table with proportional columns
        self.tree = ttk.Treeview(self.results_frame, columns=("Sl.No", "Model", "RMS Error (mV)", "MAE (mV)", "MAPE (%)", "R²", "Plot"), show="headings", yscrollcommand=self.scrollbar.set)
        # Configure column widths and stretch
        self.tree.column("Sl.No", width=50, stretch=False)
        self.tree.column("Model", width=170, stretch=True)  # This column will expand to fill extra space
        self.tree.column("RMS Error (mV)", width=130, stretch=False)
        self.tree.column("MAE (mV)", width=80, stretch=False)
        self.tree.column("MAPE (%)", width=80, stretch=False)
        self.tree.column("R²", width=80, stretch=False)
        self.tree.column("Plot", width=70, stretch=False)

        # Pack the Treeview
        self.tree.pack(fill="both", expand=True)


        # Configure the scrollbar
        self.scrollbar.config(command=self.tree.yview)

        # Set headings
        for col in self.tree["columns"]:
            self.tree.heading(col, text=col)

        # Status Label
        self.status_label = tk.Label(main_frame, text="Preparing test data...", font=("Helvetica", 12))
        self.status_label.pack(pady=10)

        # Open Results Folder Button (hidden initially)
        self.open_results_button = tk.Button(main_frame, text="Open Results Folder", command=self.open_results_folder)
        self.open_results_button.pack(pady=10)
        self.open_results_button.pack_forget()  # Hide initially

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
        self.testing_manager.start_testing(self.queue, self.update_status)

        # Process the testing tasks concurrently and display results
        while True:
            try:
                result = self.queue.get_nowait()
                self.add_result_row(result)
            except Empty:
                time.sleep(1)
                # Break the loop if the queue is empty and there are no more tasks
                if not self.testing_manager.testing_thread.is_alive():
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
        model_name = result.get("model")
        rms_error = result.get("rms_error_mv")
        mae = result.get("mae_mv")
        mape = result.get("mape")
        r2 = result.get("r2")

        # Insert the result row into the treeview
        row_id = self.tree.insert("", "end", values=(len(self.tree.get_children()), model_name, rms_error, mae, mape, r2))

        # Create the Plot button for this row
        plot_button = tk.Button(self.tree, text="Plot", command=lambda m=model_name: self.plot_result(m))

        # Place the button in the correct cell
        self.tree.set(row_id, "Plot", "Plot")
        self.tree.tag_bind(row_id, '<Button-1>', lambda e: self.plot_result(model_name))


    def open_results_folder(self):
        # Open the results folder using the default file explorer
        results_folder = self.job_manager.get_test_results_folder()
        if os.path.exists(results_folder):
            os.startfile(results_folder)

if __name__ == "__main__":
    root = tk.Tk()
    gui = VEstimTestingGUI(root)
    root.mainloop()
