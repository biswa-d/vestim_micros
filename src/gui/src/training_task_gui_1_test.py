import tkinter as tk
import json, torch
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from queue import Queue, Empty
from threading import Thread
import time
from src.services.model_training.src.training_task_service_test import TrainingTaskService
from src.gateway.src.training_task_manager_test import TrainingTaskManager
from src.gateway.src.training_setup_manager_test import VEstimTrainingSetupManager
from src.gateway.src.job_manager import JobManager
from src.gui.src.testing_gui_2_test import VEstimTestingGUI

class VEstimTrainingTaskGUI:
    def __init__(self, master, task_list, params, job_manager):
        self.master = master
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)  # Bind the window close event
        self.training_task_manager = TrainingTaskManager()
        self.training_setup_manager = VEstimTrainingSetupManager()
        self.training_service = TrainingTaskService()
        self.job_manager = JobManager()

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
        self.training_process_stopped = False
        # Add this flag to avoid multiple completions
        self.task_completed_flag = False

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

        # Clear the plot to ensure no residual data from the previous task
        self.clear_plot()

        # Set window title and size
        self.master.title(f"VEstim - Training Task {self.current_task_index + 1}")
        self.master.geometry("900x600")
        self.master.minsize(900, 600)

        # Create main frame
        main_frame = tk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        main_frame.pack_propagate(False)

        # Title Label
        title_label = tk.Label(main_frame, text="Training LSTM Model with Hyperparameters", font=("Helvetica", 12, "bold"), fg="#006400")
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
        print(f"Entering setup_time_and_plot with task {self.current_task_index + 1}: {task}")
        self.setup_time_and_plot(main_frame, task)
        self.setup_log_window(main_frame, task)

        # Stop button
        self.stop_button = tk.Button(main_frame, text="Stop Training", command=self.stop_training, bg="red", fg="white")
        self.stop_button.pack(pady=8)
        # Progress Label - initially hidden
        self.progress_label = tk.Label(
            main_frame,
            text="Processing...",
            font=("Helvetica", 10),
            bg="#e6f7ff",  # Light blue background
            fg="black"
        )
        self.progress_label.pack(pady=10)
        self.progress_label.pack_forget()  # Initially hidden

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

        self.ax.set_title(
            "Training and Validation Loss",
            fontsize=12,  # Font size
            fontweight='normal',  # Font weight
            color='#0f0c0c',  # Title color
            pad=6  # Padding between the title and the plot
        )
              

        # Initialize the plot lines with empty data
        self.train_line, = self.ax.plot([], [], label='Train Loss')
        self.valid_line, = self.ax.plot([], [], label='Validation Loss')
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
        self.log_text.insert(tk.END, f"Repetition: {task['hyperparams']['REPETITIONS']}\n")
        self.log_text.see(tk.END)

    def start_task_processing(self):
        if getattr(self, 'training_process_stopped', False):
            print("Training process has been stopped. No further tasks will be processed.")
            self.status_label.config(text="Training process has been stopped.", fg="red")
            self.show_proceed_to_testing_button()  # Show the proceed to testing button if the process is stopped
            return

        # Proceed with the task processing as usual
        self.status_label.config(
            text=f"Task {self.current_task_index + 1}/{len(self.task_list)} is running. "
                f"LSTM model being trained with hyperparameters...",
            fg="#004d99",  # Dark blue text color for emphasis
            font=("Helvetica", 10, "bold")
        )
        self.master.update()  # Ensure the message is displayed immediately

        # Start processing tasks sequentially
        self.start_time = time.time()
        self.clear_plot()

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
            
            # Continue updating the timer as long as the task is running
            if self.master.winfo_exists():
                self.master.after(1000, self.update_elapsed_time)
        else:
            # When the timer stops, it will not continue to update, but the last time will be displayed
            elapsed_time = time.time() - self.start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.time_value_label.config(text=f"{int(hours):02}h:{int(minutes):02}m:{int(seconds):02}s")


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
            learning_rate = progress_data.get('learning_rate', None)  # Default to 0.0 if not present
            best_val_loss = progress_data.get('best_val_loss', None)  # Default to 0.0 if not present
            # If learning_rate is None, print a warning
            if learning_rate is None:
                print("Warning: learning_rate is not found in progress_data")


            # Append new data to the existing lists
            self.train_loss_values.append(train_loss)
            self.valid_loss_values.append(val_loss)
            self.valid_x_values.append(epoch)

            # Use the correct parameter for the current task
            max_epochs = int(self.task_list[self.current_task_index]['hyperparams']['MAX_EPOCHS'])

            # Update the plot lines with the new data
            self.train_line.set_data(self.valid_x_values, self.train_loss_values)
            self.valid_line.set_data(self.valid_x_values, self.valid_loss_values)

            # Adjust y-axis limits dynamically based on the new data
            self.ax.set_ylim(min(min(self.train_loss_values), min(self.valid_loss_values)) * 0.9,
                            max(max(self.train_loss_values), max(self.valid_loss_values)) * 1.1)

            # Ensure x-limits remain consistent
            self.ax.set_xlim(1, max_epochs)

            # Redraw the plot
            self.canvas.draw()

            # Update the log text widget with the new progress data
            # log_message = f"\tEpoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, ΔT: {delta_t_valid:.2f}s, LR: {learning_rate}\n"
            
            # Create a tag for bold text that can be applied to values
            self.log_text.tag_configure('bold', font=('Helvetica', 9, 'bold'))
            self.log_text.tag_configure('b_bold', font=('Helvetica', 9, 'normal'), foreground='#310847') # Dark purple color, change if required


            # Insert the text with the bold tag applied to the values
            self.log_text.insert(tk.END, f" Epoch: ", '')
            self.log_text.insert(tk.END, f"{epoch}, ", 'bold')
            self.log_text.insert(tk.END, f" Train Loss: ", '')
            self.log_text.insert(tk.END, f"{train_loss:.4f}, ", 'bold')
            self.log_text.insert(tk.END, f" Val Loss: ", '')
            self.log_text.insert(tk.END, f"{val_loss:.4f}, ", 'bold')
            self.log_text.insert(tk.END, f" Best Val Loss: ", '')
            self.log_text.insert(tk.END, f"{best_val_loss:.4f}, ", 'bold')
            self.log_text.insert(tk.END, f" Time since last Val (ΔT): ", '')
            self.log_text.insert(tk.END, f"{delta_t_valid:.2f}s, ", 'bold')
            self.log_text.insert(tk.END, f" LR: ", '')
            self.log_text.insert(tk.END, f"{learning_rate:.1e}\n", 'bold')

            # Ensure the log is scrolled to the end
            self.log_text.see(tk.END)

        self.master.update()  # Ensure all updates are reflected immediately


    def clear_plot(self):
        """Clear the existing plot and reinitialize plot lines for a new task."""
        self.train_loss_values = []
        self.valid_loss_values = []
        self.valid_x_values = []

        if hasattr(self, 'ax'):  # Ensure ax exists (important for the first task)
            # Clear the plot and reset title, labels, and lines
            self.ax.clear()
            self.ax.set_title("Training and Validation Loss")
            self.ax.set_xlabel("Epoch")
            self.ax.set_ylabel("Loss [% RMSE]")

            # Reinitialize the plot lines after clearing the plot
            self.train_line, = self.ax.plot([], [], label='Train Loss')
            self.valid_line, = self.ax.plot([], [], label='Validation Loss')
            self.ax.legend()

            # Redraw the canvas with the cleared plot
            self.canvas.draw()
    
    def stop_training(self):
        print("Stop training button clicked")

        # Stop the timer
        self.timer_running = False

        # Send stop request to the task manager
        self.training_task_manager.stop_task()
        print("Stop request sent to training task manager")

        # Immediate GUI update to reflect the stopping state
        self.status_label.config(text="Stopping Training...", fg="red")
        self.master.update_idletasks()

        # Disable stop button to avoid multiple presses
        self.stop_button.config(text="Stopping...", state=tk.DISABLED, bg="grey")

        # Set flag to prevent further tasks
        self.training_process_stopped = True
        print(f"Training process stopped flag is now {self.training_process_stopped}")

        # Check if the training thread has finished
        self.master.after(100, self.check_if_stopped)


    def check_if_stopped(self):
        if self.training_thread and self.training_thread.is_alive():
            # Keep checking until the thread has stopped
            self.master.after(100, self.check_if_stopped)
        else:
            # Once the thread is confirmed to be stopped, proceed to task completion
            print("Training thread has stopped.")
            
            # Update status to indicate training has stopped
            self.status_label.config(text="Training stopped.", fg="red")
            self.master.update_idletasks()

            # Call task_completed() after confirming the thread has stopped
            if not self.task_completed_flag:
                print("Calling task_completed() after training thread has stopped.")
                self.task_completed()
            else:
                print("task_completed() was already called, skipping.")
            # Display the proceed to testing button
            # self.show_proceed_to_testing_button()



    def task_completed(self):
        print("Entering task_completed() method.")
        if self.task_completed_flag:
            return  # Exit if this method has already been called for this task
        self.task_completed_flag = True  # Set the flag to True on the first call

        self.timer_running = False

        if self.master.winfo_exists():  # Check if the window still exists
            if getattr(self, 'training_process_stopped', False):
                print(f"Training process stopped flag: {self.training_process_stopped}")
                print("Training process was stopped early.")
                self.status_label.config(text="Training stopped early. Saving model to task folder...", fg="red")
                self.show_proceed_to_testing_button()
                return

            # Check if there are more tasks to process
            if self.current_task_index < len(self.task_list) - 1:
                print(f"Completed task {self.current_task_index + 1}/{len(self.task_list)}.")
                self.current_task_index += 1
                self.task_completed_flag = False  # Reset the flag for the next task
                self.build_gui(self.task_list[self.current_task_index])
                self.start_task_processing()
            else:
                total_training_time = time.time() - self.start_time
                total_hours, total_remainder = divmod(total_training_time, 3600)
                total_minutes, total_seconds = divmod(total_remainder, 60)
                formatted_total_time = f"{int(total_hours):02}h:{int(total_minutes):02}m:{int(total_seconds):02}s"

                self.static_text_label.config(text="Total Training Time:")
                self.time_value_label.config(text=f"{formatted_total_time}")

                self.status_label.config(text="All Training Tasks Completed!")
                self.show_proceed_to_testing_button()
        else:
            print("Task completed method was called after the window was destroyed.")


    def save_model(self, task):
        """Save the trained model to disk using the task's parameters."""
        model_path = task.get('model_path', None)
        if model_path is None:
            raise ValueError("Model path not found in task.")

        model = task.get('model', None)
        if model is None:
            raise ValueError("No model instance found in task.")

        # Save the model's state dictionary to the specified path
        print(f"Saving model to {model_path}...")
        torch.save(model.state_dict(), model_path)

        # Save the hyperparameters associated with the model
        hyperparams_path = model_path + '_hyperparams.json'
        with open(hyperparams_path, 'w') as f:
            json.dump(model.hyperparams, f, indent=4)
        
        print(f"Model saved successfully at {model_path} and hyperparameters at {hyperparams_path}.")

 
    def on_closing(self):
        # Handle the window close event
        if self.training_thread and self.training_thread.is_alive():
            print("Stopping training before closing...")
            self.stop_training()  # Stop the training thread
            self.master.after(100, self.wait_for_thread_to_stop)
        else:
            self.master.destroy()  # Close the window

    def wait_for_thread_to_stop(self):
        if self.training_thread and self.training_thread.is_alive():
            # Continue checking until the thread has stopped
            self.master.after(100, self.wait_for_thread_to_stop)
        else:
            # Once the thread is confirmed to be stopped
            print("Training thread has stopped, now closing the window.")
            self.master.destroy()  # Close the window

    def show_proceed_to_testing_button(self):
        if hasattr(self, 'proceed_button'):
            return  # Exit if the button has already been created
        self.stop_button.pack_forget()
        self.proceed_button = tk.Button(self.master, text="Proceed to Testing", font=("Helvetica", 12, "bold"), fg="white", bg="green", command=self.transition_to_testing_gui)
        self.proceed_button.pack(pady=20)
    
    def transition_to_testing_gui(self):
        self.master.destroy()  # Close the training window
        root = tk.Tk()  # Create a new root for the testing GUI
        VEstimTestingGUI(root)  # Initialize the testing GUI
        root.mainloop()  # Start the main loop for the testing GUI



if __name__ == "__main__":
    root = tk.Tk()
    # Replace `task_list`, `params`, and `job_manager` with actual instances
    gui = VEstimTrainingTaskGUI(root, task_list, params, job_manager)
    root.mainloop()
