import tkinter as tk
from threading import Thread
import time
from src.gateway.src.training_manager_test import VEstimTrainingManager
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

        # Initialize the Training Manager
        self.training_manager = VEstimTrainingManager(self.update_status)
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
        title_label.pack(pady=(10, 20))  # Increased padding between title and hyperparameters
        title_label.config(underline=0)

        # Frame for hyperparameter labels with better padding
        self.hyperparam_frame = tk.Frame(main_frame)
        self.hyperparam_frame.pack(fill=tk.X, pady=(0, 20))  # Added padding below hyperparameters

        # Display placeholders for hyperparameter values in two columns
        self.display_hyperparameters(self.params)

        # Initialize status label with increased font size and padding
        self.status_label = tk.Label(main_frame, text="Setting up training...", fg="green", font=("Helvetica", 12, "bold"))
        self.status_label.pack(pady=(10, 10))

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

            self.training_manager.setup_training()  # Automatically updates the status as it progresses

            self.update_status("Task info and metadata being saved\nin task folders\n", self.job_manager.get_job_folder())
            time.sleep(2)  # Optional delay before transitioning

            self.timer_running = False  # Stop the timer before transitioning
            self.master.after(0, self.transition_to_training_gui)  # Transition to the training GUI

        self.setup_thread = Thread(target=run_setup)
        self.setup_thread.setDaemon(True)
        self.setup_thread.start()

        self.update_elapsed_time()

    def update_status(self, message, path=""):
        # Break the path into a smaller font size
        path_label = f"\n{path}"
        formatted_message = f"{message}{path_label}"

        # Update the status label with the formatted message
        self.status_label.config(text=formatted_message, font=("Helvetica", 10, "italic"))  # Smaller and italic font for path


    def transition_to_training_gui(self):
        # Clear the existing content from the master window
        self.master.destroy()  # Close the setup window
        root = tk.Tk()  # Create a new root for the training GUI
        from training_gui import VEstimTrainingGUI  # Import the Training GUI class
        VEstimTrainingGUI(root, self.params, self.job_manager)  # Initialize the Training GUI
        root.mainloop()  # Start the main loop for the training GUI

    def display_hyperparameters(self, params):
        # Clear previous widgets in the hyperparam_frame
        for widget in self.hyperparam_frame.winfo_children():
            widget.destroy()

        # Number of columns for parameters and values
        num_columns = 2
        items = list(params.items())
        
        # Calculate number of rows needed
        num_rows = (len(items) + num_columns - 1) // num_columns
        
        # Create a grid layout for parameters and values
        for i in range(num_rows):
            for j in range(num_columns):
                index = i + j * num_rows
                if index < len(items):
                    param, value = items[index]
                    label_text = self.param_labels.get(param, param)  # Get the user-friendly label or fallback to the key
                    
                    # Label for the parameter name
                    param_label = tk.Label(self.hyperparam_frame, text=f"{label_text}: ", font=("Helvetica", 12), anchor="w")
                    param_label.grid(row=i, column=j*2, sticky="w", padx=(14, 6), pady=2)
                    
                    # Label for the parameter value
                    value_label = tk.Label(self.hyperparam_frame, text=f"{value}", font=("Helvetica", 12, "bold"), fg="#004d99", anchor="w")
                    value_label.grid(row=i, column=j*2+1, sticky="w", padx=(0, 10), pady=4)
        
        # Adjust column weights to ensure labels are left-aligned and spaced properly
        for col in range(num_columns * 2):
            self.hyperparam_frame.grid_columnconfigure(col, weight=1)

    
    # def display_hyperparameters(self, params):
    #     # Clear previous widgets in the hyperparam_frame
    #     for widget in self.hyperparam_frame.winfo_children():
    #         widget.destroy()

    #     # Number of columns for parameters and values
    #     num_columns = 2
    #     items = list(params.items())
        
    #     # Calculate number of rows needed
    #     num_rows = (len(items) + num_columns - 1) // num_columns
        
    #     # Create a grid layout for parameters and values
    #     for i in range(num_rows):
    #         for j in range(num_columns):
    #             index = i + j * num_rows
    #             if index < len(items):
    #                 param, value = items[index]
    #                 label_text = self.param_labels.get(param, param)  # Get the user-friendly label or fallback to the key

    #                 # Create a frame to contain the parameter label and value, adding border and padding
    #                 frame = tk.Frame(self.hyperparam_frame, bd=1, relief="solid", padx=5, pady=5, bg="#f2f2f2")
    #                 frame.grid(row=i, column=j, padx=10, pady=5, sticky="w")

    #                 # Label for the parameter name inside the frame
    #                 param_label = tk.Label(frame, text=f"{label_text}: ", font=("Helvetica", 12), anchor="w", bg="#f2f2f2")
    #                 param_label.pack(side="left", padx=(5, 2))

    #                 # Label for the parameter value inside the frame
    #                 value_label = tk.Label(frame, text=f"{value}", font=("Helvetica", 12, "bold"), fg="#004d99", anchor="w", bg="#f2f2f2")
    #                 value_label.pack(side="left", padx=(2, 5))

    #     # Adjust column weights to ensure labels are left-aligned and spaced properly
    #     for col in range(num_columns):
    #         self.hyperparam_frame.grid_columnconfigure(col, weight=1)


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
