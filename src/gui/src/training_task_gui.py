import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from queue import Queue, Empty
from threading import Thread
import time
from src.gateway.src.training_task_manager import TrainingTaskManager
from src.gateway.src.job_manager import JobManager
# from src.gui.src.testing_gui_test import VEstimTestingGui

class VEstimTrainingTaskGUI:
    def __init__(self, master, task_list, params, job_manager):
        self.master = master
        self.training_task_manager = TrainingTaskManager()

        # Store the task list, params, and job_manager
        self.task_list = task_list
        self.params = params
        self.job_manager = job_manager

        # Initialize variables
        self.train_loss_values = []
        self.valid_loss_values = []
        self.valid_x_values = []
        self.start_time = None
        self.queue = Queue()
        self.timer_running = True

        # Dictionary for displaying labels
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

        # Start processing the first task
        self.current_task_index = 0
        self.build_gui(self.task_list[self.current_task_index])
        self.start_task_processing()

    def build_gui(self, task):
        # Clear existing content
        for widget in self.master.winfo_children():
            widget.destroy()

        # Set window title and size
        self.master.title(f"VEstim - Training Task {self.current_task_index + 1}")
        self.master.geometry("900x600")
        self.master.minsize(900, 600)

        # Create main frame
        main_frame = tk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        main_frame.pack_propagate(False)

        # Title Label
        title_label = tk.Label(main_frame, text="Training LSTM Model with Hyperparameters", font=("Helvetica", 12, "bold"))
        title_label.pack(pady=5)

        # Hyperparameters Frame
        self.hyperparam_frame = tk.Frame(main_frame)
        self.hyperparam_frame.pack(fill=tk.X, pady=5)

        # Display hyperparameters from the task
        self.display_hyperparameters(task['hyperparams'])

        # Status Label
        self.status_label = tk.Label(main_frame, text="Starting training...", fg="green", font=("Helvetica", 10, "bold"))
        self.status_label.pack(pady=5)

        # Time Frame, Plot Setup, and Log Window
        self.setup_time_and_plot(main_frame, task)
        self.setup_log_window(main_frame, task)

        # Stop button
        self.stop_button = tk.Button(main_frame, text="Stop Training", command=self.stop_training, bg="red", fg="white")
        self.stop_button.pack(pady=10)

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

    def setup_time_and_plot(self, main_frame, task):
        # Debugging statement to check the structure of the task
        print(f"Current task: {task}")
        print(f"Hyperparameters in the task: {task['hyperparams']}")

        # Time Frame
        time_frame = tk.Frame(main_frame)
        time_frame.pack(pady=5)

        # Label for static text
        self.static_text_label = tk.Label(time_frame, text="Time Since Training Started:", fg="blue", font=("Helvetica", 10))
        self.static_text_label.pack(side=tk.LEFT)

        # Label for the dynamic time value in bold
        self.time_value_label = tk.Label(time_frame, text="00h:00m:00s", fg="blue", font=("Helvetica", 10, "bold"))
        self.time_value_label.pack(side=tk.LEFT)

        # Ensure 'MAX_EPOCHS' and 'ValidFrequency' are integers
        max_epochs = int(task['hyperparams']['MAX_EPOCHS'])
        valid_frequency = int(task['hyperparams']['ValidFrequency'])

        # Setting up Matplotlib figure for loss plots
        fig = Figure(figsize=(6, 2.5), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.set_xlabel("Epoch", labelpad=0)
        self.ax.set_ylabel("Loss [% RMSE]")
        self.ax.set_xlim(1, max_epochs)

        # Set x-ticks to ensure a maximum of 10 parts or based on validation frequency
        max_ticks = 10
        if max_epochs <= max_ticks:
            xticks = list(range(1, max_epochs + 1))
        else:
            xticks = list(range(1, max_epochs + 1, max(1, max_epochs // max_ticks)))
        
        # Ensure the last epoch is included
        if max_epochs not in xticks:
            xticks.append(max_epochs)
        
        self.ax.set_xticks(xticks)
        self.ax.set_xticklabels(xticks, rotation=45, ha="right")  # Rotate labels for better readability
        
        # # Generate x-ticks based on the max epochs and validation frequency
        # xticks = list(range(1, max_epochs + 1, valid_frequency))
        # if max_epochs not in xticks:
        #     xticks.append(max_epochs)
        # self.ax.set_xticks(xticks)

        self.ax.set_title("Training and Validation Loss")
        self.ax.legend(["Train Loss", "Validation Loss"])

        # Initialize the plot lines
        self.ax.plot([], [], label='Train Loss')
        self.ax.plot([], [], label='Validation Loss')
        self.ax.legend()

        # Attach the Matplotlib figure to the Tkinter frame
        self.canvas = FigureCanvasTkAgg(fig, master=main_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.ax.margins(y=0.1)
        fig.subplots_adjust(bottom=0.2)

    def setup_log_window(self, main_frame, task):
        # Rolling window for displaying detailed logs
        self.log_text = tk.Text(main_frame, height=1, wrap=tk.WORD)
        self.log_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # self.log_text.insert(tk.END, "Initial Parameters:\n")
        self.log_text.insert(tk.END, f"Repetition: 1/{task['hyperparams']['REPETITIONS']}, "
                                    f"Epoch: 0, Train Loss: N/A, Validation Error: N/A\n")
        self.log_text.see(tk.END)

    def start_task_processing(self):
        # Update the status label to show that the task is starting
        self.status_label.config(
            text=f"Task {self.current_task_index + 1}/{len(self.task_list)} is running. "
                f"LSTM model being trained with hyperparameters...",
            fg="#004d99",  # Dark blue text color for emphasis
            font=("Helvetica", 10, "bold")
        )
        self.master.update()  # Ensure the message is displayed immediately

        # Start processing tasks sequentially
        self.start_time = time.time()

        def run_training_task(task):
            # Process the current task using the TrainingTaskManager
            self.training_task_manager.process_task(task, self.queue, self.update_gui_after_epoch)

        self.training_thread = Thread(target=run_training_task, args=(self.task_list[self.current_task_index],))
        self.training_thread.setDaemon(True)
        self.training_thread.start()

        self.update_elapsed_time()
        self.process_queue()

    def update_elapsed_time(self):
        if self.timer_running:
            elapsed_time = time.time() - self.start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.time_value_label.config(text=f"{int(hours):02}h:{int(minutes):02}m:{int(seconds):02}s")
            self.master.after(1000, self.update_elapsed_time)

    def process_queue(self):
        try:
            while True:
                progress_data = self.queue.get_nowait()
                if 'task_error' in progress_data:
                    self.handle_error(progress_data['task_error'])
                    break  # Stop further processing
                elif 'task_completed' in progress_data:
                    self.task_completed()
                    break  # Exit the loop once the task is completed
        except Empty:
            self.master.after(100, self.process_queue)

    def handle_error(self, error_message):
        """Display an error message in the GUI."""
        self.status_label.config(text=f"Error occurred: {error_message}", fg="red")
        self.log_text.insert(tk.END, f"Error: {error_message}\n")
        self.log_text.see(tk.END)
        self.stop_button.pack_forget()

    def update_gui_after_epoch(self, progress_data):
        # Combine task index and dynamic status for the status label
        task_info = f"Task {self.current_task_index + 1}/{len(self.task_list)}"

        # Always show the task info, and append the status if available
        if 'status' in progress_data:
            self.status_label.config(
                text=f"{task_info} - {progress_data['status']}",
                fg="#004d99",  # Dark blue text color for emphasis
                font=("Helvetica", 10, "bold")
            )
        else:
            self.status_label.config(
                text=f"{task_info} - LSTM model being trained with hyperparameters...",
                fg="#004d99",  # Dark blue text color for emphasis
                font=("Helvetica", 10, "bold")
            )

        # Now handle epoch-based updates as before
        if 'epoch' in progress_data:
            epoch = progress_data['epoch']
            train_loss = progress_data['train_loss']
            val_loss = progress_data['val_loss']
            delta_t_valid = progress_data['delta_t_valid']

            # Append new data to the existing lists
            self.train_loss_values.append(train_loss)
            self.valid_loss_values.append(val_loss)
            self.valid_x_values.append(epoch)

            # Clear the plot and re-plot with the new data
            self.ax.clear()
            self.ax.set_title("Training and Validation Loss")
            self.ax.set_xlabel("Epoch")
            self.ax.set_ylabel("Loss [% RMSE]")
            self.ax.plot(self.valid_x_values, self.train_loss_values, label='Train Loss')
            self.ax.plot(self.valid_x_values, self.valid_loss_values, label='Validation Loss')
            self.ax.legend()
            self.canvas.draw()

            # Update the log text widget with the new progress data
            log_message = f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, ΔT: {delta_t_valid:.2f}s\n"
            self.log_text.insert(tk.END, log_message)
            self.log_text.see(tk.END)
        
        self.master.update()  # Ensure all updates are reflected immediately


    def stop_training(self):
        if self.training_thread.is_alive():
            self.timer_running = False
            self.training_task_manager.stop_task()  # Call to stop the task

            self.stop_button.pack_forget()
            self.status_label.config(text="Training Stopped! Saving Model...", fg="red")

            # Proceed to saving and completing the task
            self.master.after(1000, self.task_completed)

    def task_completed(self):
        # Called when a task is completed
        self.timer_running = False

        # Update GUI for completed task
        self.status_label.config(text=f"Task {self.current_task_index + 1} Completed! Saving model to task folder...")

        # Pause for a moment to allow the user to read the completion status
        self.master.update()  # Force update to display the message
        time.sleep(3)  # Pause for 3 seconds

        # Proceed to the next task or finish
        if self.current_task_index < len(self.task_list) - 1:
            self.current_task_index += 1
            self.status_label.config(text="Starting the next task...")
            self.master.update()
            self.build_gui(self.task_list[self.current_task_index])
            self.start_task_processing()
        else:
            self.status_label.config(text="All Training Tasks Completed!")
            self.show_proceed_to_testing_button()

    def show_proceed_to_testing_button(self):
        # Create a button to proceed to testing
        self.proceed_button = tk.Button(self.master, text="Proceed to Testing", font=("Helvetica", 12, "bold"), fg="white", bg="green", command=self.transition_to_testing_gui)
        self.proceed_button.pack(pady=20)  # Adjust padding as necessary

    def transition_to_testing_gui(self):
        # Destroy the current window and move to the testing GUI
        self.master.destroy()
        root = tk.Tk()
        # VEstimTestingGui(root)  # Assuming VEstimTestingGui is the class for the testing GUI
        root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    # Replace `task_list`, `params`, and `job_manager` with actual instances
    gui = VEstimTrainingTaskGUI(root, task_list, params, job_manager)
    root.mainloop()
