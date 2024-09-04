import tkinter as tk
from threading import Thread
import time
from src.gateway.src.training_setup_manager import VEstimTrainingSetupManager
from src.gui.src.training_task_gui import VEstimTrainingTaskGUI
from src.gateway.src.job_manager import JobManager

class VEstimTrainSetupGUI:
    def __init__(self, master, params, job_manager):
        self.master = master
        self.params = params
        self.job_manager = job_manager
        self.timer_running = True  # Initialize the timer_running flag
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

        # Initialize the Training Setup Manager
        self.training_setup_manager = VEstimTrainingSetupManager(self.update_status)
        self.build_gui()

    def build_gui(self):
        # Clear the existing content from the master window
        for widget in self.master.winfo_children():
            widget.destroy()

        # Update the window title and show initial message
        self.master.title("VEstim - Setting Up Training")
        self.master.geometry("900x600")
        self.master.minsize(900, 600)

        # Create main frame
        main_frame = tk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        main_frame.pack_propagate(False)

        # Title Label with formatting and added padding
        title_label = tk.Label(main_frame, text="Building LSTM Models and Training Tasks\nwith Hyperparameter Set",
                            font=("Helvetica", 12, "bold"), fg="#3a3a3a", anchor='center', justify='center')
        title_label.pack(pady=(20, 40))  # Increased padding between title and hyperparameters

        # Frame for hyperparameter labels with better padding
        self.hyperparam_frame = tk.Frame(main_frame)
        self.hyperparam_frame.pack(fill=tk.X, padx=(20, 20), pady=(0, 20))  # Added padding below hyperparameters

        # Display placeholders for hyperparameter values in two columns
        self.display_hyperparameters(self.params)

        # Initialize status label with increased font size and padding
        self.status_label = tk.Label(main_frame, text="Setting up training...", fg="green", font=("Helvetica", 12, "bold"))
        self.status_label.pack(pady=(20, 10))

        # Create a frame to hold both labels
        time_frame = tk.Frame(main_frame)
        time_frame.pack(pady=5)

        # Label for static text
        self.static_text_label = tk.Label(time_frame, text="Time Since Setup Started:", fg="blue", font=("Helvetica", 10))
        self.static_text_label.pack(side=tk.LEFT)

        # Label for the dynamic time value in bold with padding
        self.time_value_label = tk.Label(time_frame, text="00h:00m:00s", fg="blue", font=("Helvetica", 12, "bold"))
        self.time_value_label.pack(side=tk.LEFT, padx=(5, 0))
        # Start the setup process
        self.start_setup()

    def start_setup(self):
        self.start_time = time.time()

        def run_setup():
            self.update_status("Building models and data loaders...")

            # Run the training setup
            self.training_setup_manager.setup_training()

            # Get the number of training tasks created
            task_count = len(self.training_setup_manager.training_tasks)

            # Update the status with the task count
            self.update_status("Task summary saved in the job folder\n", 
                            self.job_manager.get_job_folder(), task_count)

            time.sleep(2)  # Optional delay before enabling the button

            self.timer_running = False  # Stop the timer after task creation is complete
            self.show_proceed_button()  # Show the proceed button after the setup is complete

        self.setup_thread = Thread(target=run_setup)
        self.setup_thread.setDaemon(True)
        self.setup_thread.start()

        self.update_elapsed_time()

    def update_status(self, message, path="", task_count=None):
        if task_count is not None:
            task_message = f"{task_count} training tasks created,\n"
        else:
            task_message = ""

        if path:
            formatted_message = f"{task_message}{message}\n{path}"
        else:
            formatted_message = f"{task_message}{message}"

        self.status_label.config(text=formatted_message, font=("Helvetica", 10, "italic"))

    def show_proceed_button(self):
        proceed_button = tk.Button(self.master, text="Proceed to Training", font=("Helvetica", 12, "bold"), fg="white", bg="green", command=self.transition_to_training_gui)
        proceed_button.pack(pady=20)

    def transition_to_training_gui(self):
        self.master.destroy()  # Close the setup window
        root = tk.Tk()  # Create a new root for the training GUI
        task_list = self.training_setup_manager.get_task_list()  # Get the task list
        VEstimTrainingTaskGUI(root, task_list, self.params, self.job_manager)  # Pass the task list, params, and job_manager
        root.mainloop()  # Start the main loop for the training GUI


    def display_hyperparameters(self, params):
        # Clear the previous widgets in the hyperparam_frame
        for widget in self.hyperparam_frame.winfo_children():
            widget.destroy()

        # Set number of columns and rows based on parameter count
        num_columns = 4
        items = list(params.items())
        num_rows = (len(items) + 1) // 2  # Number of rows to display the hyperparameters

        # Iterate through the rows and columns to display hyperparameters
        for i in range(num_rows):
            for j in range(2):  # Two columns of parameters
                index = i * 2 + j
                if index < len(items):
                    param, value = items[index]
                    label_text = self.param_labels.get(param, param)

                    # Handle the case where the value contains multiple items (e.g., 100,200,300,400, etc.)
                    value_str = str(value)
                    if "," in value_str:
                        values = value_str.split(",")  # Split the values
                        if len(values) > 2:  # Show only first two values and append '...'
                            display_value = f"{values[0]},{values[1]},..."
                        else:
                            display_value = value_str  # Show all values if 2 or less
                    else:
                        display_value = value_str  # If it's a single value

                    # Create and display parameter label
                    param_label = tk.Label(self.hyperparam_frame, text=f"{label_text}: ", font=("Helvetica", 10), anchor="w", 
                                        bg="#f0f0f0", bd=0.5, relief="solid", padx=5, pady=2)
                    param_label.grid(row=i, column=j * 2, sticky="w", padx=(10, 5), pady=4, ipadx=4, ipady=5)
                    param_label.config(width=20)

                    # Create and display value label (truncated if needed)
                    value_label = tk.Label(self.hyperparam_frame, text=f"{display_value}", font=("Helvetica", 10, "bold"), fg="#005878", 
                                        anchor="w", bg="#f0f0f6", bd=0.5, relief="solid", padx=5, pady=2)
                    value_label.grid(row=i, column=j * 2 + 1, sticky="w", padx=(5, 10), pady=4, ipadx=4, ipady=5)
                    value_label.config(width=10)

        # Configure grid columns and rows to ensure proper resizing and alignment
        self.hyperparam_frame.grid_columnconfigure(0, weight=2)
        self.hyperparam_frame.grid_columnconfigure(1, weight=1)
        self.hyperparam_frame.grid_columnconfigure(2, weight=2)
        self.hyperparam_frame.grid_columnconfigure(3, weight=1)

        for i in range(num_rows):
            self.hyperparam_frame.grid_rowconfigure(i, weight=1)


    def update_elapsed_time(self):
        if self.timer_running:
            elapsed_time = time.time() - self.start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.time_value_label.config(text=f" {int(hours):02}h:{int(minutes):02}m:{int(seconds):02}s")
            self.master.after(1000, self.update_elapsed_time)

if __name__ == "__main__":
    root = tk.Tk()
    params = {}
    job_manager = JobManager()
    gui = VEstimTrainSetupGUI(root, params, job_manager)
    root.mainloop()
