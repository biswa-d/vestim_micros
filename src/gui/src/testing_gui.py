import tkinter as tk
from tkinter import ttk
from threading import Thread
import time, os, sys
from queue import Queue, Empty
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
from tkinter.ttk import Progressbar
from src.gateway.src.testing_manager_2_test import VEstimTestingManager
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
        self.timer_running = True
        self.start_time = None
        self.testing_thread = None
        self.results_list = []  # List to store results
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
        self.tree.column("Sl.No", width=50, stretch=False, anchor="center")
        self.tree.column("Model", width=120, stretch=True, anchor="center")
        self.tree.column("RMS Error (mV)", width=120, stretch=False, anchor="center")
        self.tree.column("MAE (mV)", width=90, stretch=False, anchor="center")
        self.tree.column("MAPE (%)", width=90, stretch=False, anchor="center")
        self.tree.column("R²", width=80, stretch=False, anchor="center")
        self.tree.column("Plot", width=90, stretch=False, anchor="center")

        self.tree.pack(fill="both", expand=True)
        self.scrollbar.config(command=self.tree.yview)

        for col in self.tree["columns"]:
            self.tree.heading(col, text=col)

        # Create a status and progress section in the center
        self.status_frame = tk.Frame(main_frame)
        self.status_frame.pack(pady=10)

        # Status Label
        self.status_label = tk.Label(self.status_frame, text="Preparing test data...", font=("Helvetica", 10), anchor="center", fg="black")
        self.status_label.pack(pady=5)

        # Add a progress bar below the status label
        self.progress = Progressbar(self.status_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.pack(pady=5)

        # Button to open results folder, initially hidden
        self.open_results_button = tk.Button(main_frame, text="Open Results Folder", command=self.open_results_folder, 
                                            bg="green", fg="white", font=("Helvetica", 10, "bold"))
        self.open_results_button.pack(pady=10)
        self.open_results_button.pack_forget()  # Hide the button initially


    def display_hyperparameters(self, params):
        # Clear previous widgets in the hyperparam_frame
        for widget in self.hyperparam_frame.winfo_children():
            widget.destroy()

        # Get the parameter items (mapping them to the correct labels if necessary)
        param_items = [(self.param_labels.get(param, param), value) for param, value in params.items()]

        # Split the parameters into five columns
        columns = [param_items[i::5] for i in range(5)]  # Splits into five columns

        # Display each column with labels
        for col_num, column in enumerate(columns):
            col_frame = tk.Frame(self.hyperparam_frame)
            col_frame.grid(row=0, column=col_num, padx=5)
            for row_num, (param, value) in enumerate(column):
                # Truncate the value if it contains more than 2 items (values separated by commas)
                value_str = str(value)
                if "," in value_str:
                    values = value_str.split(",")  # Split the values by comma
                    if len(values) > 2:  # If more than 2 values, show only the first two with "..."
                        display_value = f"{values[0]},{values[1]},..."
                    else:
                        display_value = value_str  # Display the whole value if there are 2 or fewer
                else:
                    display_value = value_str  # If it's a single value

                # Create parameter label
                param_label = tk.Label(col_frame, text=f"{param}: ", font=("Helvetica", 10))  # Regular font for the label
                value_label = tk.Label(col_frame, text=f"{display_value}", font=("Helvetica", 10, "bold"))  # Bold font for value

                # Use grid to ensure both labels stay on the same line
                param_label.grid(row=row_num, column=0, sticky='w')
                value_label.grid(row=row_num, column=1, sticky='w')

        # Centering the hyperparameters table
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

        self.process_queue()  # Start processing the queue

    def process_queue(self):
        try:
            result = self.queue.get_nowait()
            print(f"Got result from queue: {result}")
            self.add_result_row(result)
            self.results_list.append(result)
        except Empty:
            print("Queue is empty, waiting for results...")

        # Update the progress bar based on the number of tasks completed
        total_tasks = len(self.testing_manager.training_setup_manager.get_task_list())
        completed_tasks = len(self.results_list)
        self.progress['value'] = (completed_tasks / total_tasks) * 100

        # Update the status label with a short message
        self.update_status(f"Completed {completed_tasks}/{total_tasks} tasks")

        if completed_tasks >= total_tasks:
            print("All tasks completed. Stopping queue processing.")
            self.timer_running = False
            self.update_status("All tests completed!")
            self.progress.pack_forget()  # Hide progress bar
            self.open_results_button.pack()
            return

        # Schedule the next queue check
        self.master.after(1000, self.process_queue)

    def update_status(self, message):
        # Update the text in the status label
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

            # Add row to Treeview with "Show Plot" in the last column
            row_id = self.tree.insert("", "end", values=(sl_no, model_name, rms_error, mae, mape, r2, "Show Plot"))

            # Bind a click event to the Treeview
            self.tree.bind("<Button-1>", self.on_tree_click)

    def on_tree_click(self, event):
        # Identify the row and column clicked
        region = self.tree.identify("region", event.x, event.y)
        if region == "cell":
            row_id = self.tree.identify_row(event.y)
            column = self.tree.identify_column(event.x)

            # Check if the 7th column ("Plot" column) is clicked
            if column == "#7":  # "#7" refers to the 7th column where "Show Plot" is located
                # Get the model name from the clicked row
                model_name = self.tree.item(row_id, 'values')[1]
                print(f"Plot clicked for model: {model_name}")

                # Call the plot method
                self.plot_model_results(model_name)

    def plot_model_results(self, model_name):
        """
        Plot the test results for a specific model by reading from the saved CSV file and provide
        an option to save the plot.

        :param model_name: The name of the model whose results will be plotted.
        """
        # Create a new window for plotting
        plot_window = tk.Toplevel(self.master)
        plot_window.title(f"Testing Results for Model: {model_name}")  # Set title for the pop-up window
        plot_window.geometry("700x500")

        # Locate the CSV file in the model-specific directory
        save_dir = self.job_manager.get_test_results_folder()  # Assuming this method returns the correct save directory
        model_dir = os.path.join(save_dir, model_name)
        result_file = os.path.join(model_dir, f"{model_name}_test_results.csv")

        # Check if the file exists
        if not os.path.exists(result_file):
            print(f"Test results file not found: {result_file}")
            return
        
        # Load the data from the CSV file
        df = pd.read_csv(result_file)

        # Extract the necessary data for plotting
        true_values = df['True Values (V)']
        predictions = df['Predictions (V)']

        # Subsampling the data points to avoid crowding
        sampling_rate = 100  # Plot every 100th data point

        # Create a matplotlib figure
        fig = Figure(figsize=(7, 4), dpi=100)
        ax = fig.add_subplot(111)

        # Plot the true values and predictions with refined styling
        ax.plot(true_values, label='True Values (V)', color='blue', marker='o', markersize=3, linestyle='-', linewidth=0.8)
        ax.plot(predictions, label='Predictions (V)', color='green', marker='x', markersize=3, linestyle='--', linewidth=0.8)

        # Customize labels, title, and legend for better aesthetics
        ax.set_xlabel('Index', fontsize=12)
        ax.set_ylabel('Voltage (V)', fontsize=12)
        ax.set_title(f"Testing Results for Model: {model_name}", fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)

        # Fine-tune grid and ticks for better readability
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=10)

        # Embed the plot in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Add a close button
        close_button = tk.Button(plot_window, text="Close", command=plot_window.destroy)
        close_button.pack(side=tk.BOTTOM, pady=10)

        # Add a button to save the plot as an image
        save_button = tk.Button(plot_window, text="Save Plot", command=lambda: self.save_plot(fig, model_name, model_dir))
        save_button.pack(side=tk.BOTTOM, pady=5)

        print(f"Plot generated for model: {model_name}")

    def save_plot(self, fig, model_name, save_dir):
        """
        Save the current plot as a PNG image.
        
        :param fig: The figure object of the plot.
        :param model_name: The name of the model being plotted.
        :param save_dir: The directory where the plot should be saved.
        """
        # Create the file path for the saved image
        plot_file = os.path.join(save_dir, f"{model_name}_test_results_plot.png")

        # Save the figure as a PNG image
        fig.savefig(plot_file, format='png')
        print(f"Plot saved as: {plot_file}")


    def open_model_directory(self, model_name):
        # This method should open the directory where the model is stored
        print(f"Open model directory for: {model_name}")

    def open_results_folder(self):
        results_folder = self.job_manager.get_test_results_folder()
        if os.path.exists(results_folder):
            os.startfile(results_folder)

if __name__ == "__main__":
    root = tk.Tk()
    gui = VEstimTestingGUI(root)
    root.mainloop()
