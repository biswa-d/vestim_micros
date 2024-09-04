import tkinter as tk
from tkinter import ttk
from threading import Thread
import time, os, sys
from queue import Queue, Empty

from src.gateway.src.testing_manager_1_test import VEstimTestingManager
from src.gateway.src.job_manager import JobManager
from src.gateway.src.training_setup_manager_test import VEstimTrainingSetupManager
from src.gateway.src.hyper_param_manager import VEstimHyperParamManager

class VEstimTestingGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("VEstim Tool - Model Testing")
        self.master.geometry("900x700")
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)  # Bind close event to on_closing method

        self.job_manager = JobManager()
        self.testing_manager = VEstimTestingManager()
        self.hyper_param_manager = VEstimHyperParamManager()
        self.training_setup_manager = VEstimTrainingSetupManager()

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
        self.testing_thread = None

        # Immediately build the GUI and start testing
        self.build_gui()
        self.start_testing()

    def on_closing(self):
        print("Attempting to close the application.")
        self.stop_testing()
        
        # Wait for the testing thread to complete
        if self.testing_thread is not None and self.testing_thread.is_alive():
            print("Waiting for testing thread to finish...")
            self.testing_thread.join()
            print("Testing thread finished.")
        
        # Properly destroy the window to end the Tkinter main loop
        self.master.destroy()
        print("Window closed. Exiting application.")
        sys.exit()  # Ensure the Python task is completely closed

    def build_gui(self):
        # Clear existing content
        for widget in self.master.winfo_children():
            widget.destroy()

        self.master.title("VEstim - Testing LSTM Models")
        self.master.geometry("900x700")
        self.master.minsize(900, 700)

        main_frame = tk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        main_frame.pack_propagate(False)

        title_label = tk.Label(main_frame, text="Testing LSTM Models", font=("Helvetica", 14, "bold"), fg="#006400")
        title_label.pack(pady=5)

        self.hyperparam_frame = tk.Frame(main_frame)
        self.hyperparam_frame.pack(fill=tk.X, pady=5)

        params = self.hyper_param_manager.get_current_params()
        self.display_hyperparameters(params)

        self.time_label = tk.Label(main_frame, text="Testing Time: 00h:00m:00s", font=("Helvetica", 10), fg="blue")
        self.time_label.pack(pady=5)

        result_summary_label = tk.Label(main_frame, text="Testing Result Summary", font=("Helvetica", 12, "bold"))
        result_summary_label.pack(pady=5)

        self.results_frame = tk.Frame(main_frame)
        self.results_frame.pack(fill=tk.BOTH, expand=True)

        self.scrollbar = tk.Scrollbar(self.results_frame)
        self.scrollbar.pack(side="right", fill="y")

        style = ttk.Style()
        style.configure("Treeview.Heading", font=("Helvetica", 10, "bold"), relief="solid", borderwidth=2)
        style.map("Treeview.Heading", background=[("!pressed", "white")], relief=[("!pressed", "groove")])
        style.configure("Treeview", font=("Helvetica", 10))

        self.tree = ttk.Treeview(self.results_frame, columns=("Sl.No", "Model", "RMS Error (mV)", "MAE (mV)", "MAPE (%)", "R²", "Plot"), show="headings", yscrollcommand=self.scrollbar.set)
        self.tree.column("Sl.No", width=50, stretch=False)
        self.tree.column("Model", width=170, stretch=True)
        self.tree.column("RMS Error (mV)", width=130, stretch=False)
        self.tree.column("MAE (mV)", width=80, stretch=False)
        self.tree.column("MAPE (%)", width=80, stretch=False)
        self.tree.column("R²", width=80, stretch=False)
        self.tree.column("Plot", width=70, stretch=False)

        self.tree.pack(fill="both", expand=True)
        self.scrollbar.config(command=self.tree.yview)

        for col in self.tree["columns"]:
            self.tree.heading(col, text=col)

        self.status_label = tk.Label(main_frame, text="Preparing test data...", font=("Helvetica", 12))
        self.status_label.pack(pady=10)

        self.open_results_button = tk.Button(main_frame, text="Open Results Folder", command=self.open_results_folder)
        self.open_results_button.pack(pady=10)
        self.open_results_button.pack_forget()

    def display_hyperparameters(self, params):
        for widget in self.hyperparam_frame.winfo_children():
            widget.destroy()

        param_items = [(self.param_labels.get(param, param), value) for param, value in params.items()]
        columns = [param_items[i::5] for i in range(5)]

        for col_num, column in enumerate(columns):
            col_frame = tk.Frame(self.hyperparam_frame)
            col_frame.grid(row=0, column=col_num, padx=5)
            for row_num, (param, value) in enumerate(column):
                param_label = tk.Label(col_frame, text=f"{param}: ", font=("Helvetica", 10))
                value_label = tk.Label(col_frame, text=f"{value}", font=("Helvetica", 10, "bold"))
                param_label.grid(row=row_num, column=0, sticky='w')
                value_label.grid(row=row_num, column=1, sticky='w')

        self.hyperparam_frame.grid_columnconfigure(0, weight=1)
        self.hyperparam_frame.grid_columnconfigure(len(columns) - 1, weight=1)

    def start_testing(self):
        self.status_label.config(text="Preparing test data...")
        self.start_time = time.time()

        self.testing_thread = Thread(target=self.run_testing)
        self.testing_thread.start()
    
    def stop_testing(self):
        print("Setting stop flag and attempting to stop testing...")
        self.stop_flag = True  # Set the stop flag
        self.testing_manager.stop_flag = True  # Also set the flag in the testing manager

        if self.testing_thread is not None and self.testing_thread.is_alive():
            print("Waiting for the testing thread to finish...")
            self.testing_thread.join(timeout=5)  # Wait for a maximum of 5 seconds for the thread to finish
            
            if self.testing_thread.is_alive():
                print("Testing thread did not finish in time. Forcing exit.")
                # Attempt to forcefully exit the thread (not recommended generally, but here as a last resort)
                # Usually, we would try to stop gracefully, but we can implement a forced exit as a last resort.

            print("Testing stopped.")

    def update_timer(self):
        if self.timer_running:
            elapsed_time = time.time() - self.start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.time_label.config(text=f"Testing Time: {int(hours):02}h:{int(minutes):02}m:{int(seconds):02}s")
            self.master.after(1000, self.update_timer)

    def run_testing(self):
        print("Starting the testing process...")
        self.update_status("Creating test tasks...")
        print("Preparing test tasks...")

        self.timer_running = True
        self.update_timer()

        print("Starting the testing manager...")
        self.testing_manager.start_testing(self.queue, self.update_status)

        print(f"Testing thread alive: {self.testing_manager.testing_thread.is_alive()}")

        while True:
            try:
                result = self.queue.get_nowait()
                print(f"Got result from queue: {result}")
                self.add_result_row(result)
            except Empty:
                print("Queue is empty, waiting for results...")
                time.sleep(1)
                if not self.testing_manager.testing_thread.is_alive():
                    print("Testing thread has finished.")
                    break

        print("Testing completed!")
        self.update_status("All tests completed!")
        self.timer_running = False
        self.open_results_button.pack()

    def update_status(self, message):
        self.status_label.config(text=message)

    def add_result_row(self, result):
        task_data = result.get('task_completed')
        if task_data:
            sl_no = task_data.get("sl_no")
            model_name = task_data.get("model")
            rms_error = f"{task_data.get('rms_error_mv', 0):.4f}"
            mae = f"{task_data.get('mae_mv', 0):.4f}"
            mape = f"{task_data.get('mape', 0):.4f}"
            r2 = f"{task_data.get('r2', 0):.4f}"

            print(f"Adding result: Sl.No: {sl_no}, Model: {model_name}, RMS Error: {rms_error}, MAE: {mae}, MAPE: {mape}, R²: {r2}")
            row_id = self.tree.insert("", "end", values=(sl_no, model_name, rms_error, mae, mape, r2))
            print(f"Row added with id: {row_id}")

            # Making the Model Name Clickable
            self.tree.tag_bind(row_id, '<Double-1>', lambda e: self.open_model_directory(model_name))


    def open_results_folder(self):
        results_folder = self.job_manager.get_test_results_folder()
        if os.path.exists(results_folder):
            os.startfile(results_folder)

if __name__ == "__main__":
    root = tk.Tk()
    gui = VEstimTestingGUI(root)
    root.mainloop()
