import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from queue import Queue, Empty
from threading import Thread
import time
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
        print(f"Entering setup_time_and_plot with task: {task}")
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


    def task_completed(self):
        self.timer_running = False

        # Check if the stop flag is set
        if getattr(self, 'training_process_stopped', False):
            print("Training process was stopped early.")
            self.status_label.config(text="Training stopped early. Saving model to task folder...", fg="red")
            self.show_proceed_to_testing_button()
            return  # Exit the task loop early

        # Move to the next task if there are more tasks to process
        if self.current_task_index < len(self.task_list) - 1:
            self.current_task_index += 1
            self.build_gui(self.task_list[self.current_task_index])
            self.start_task_processing()
        else:
            total_training_time = time.time() - self.start_time  # Total time for all tasks
            total_hours, total_remainder = divmod(total_training_time, 3600)
            total_minutes, total_seconds = divmod(total_remainder, 60)
            formatted_total_time = f"{int(total_hours):02}h:{int(total_minutes):02}m:{int(total_seconds):02}s"
            
            # Update static text label and dynamic time label for total training time
            self.static_text_label.config(text="Total Training Time:")
            self.time_value_label.config(text=f"{formatted_total_time}")
            
            self.status_label.config(text="All Training Tasks Completed!")
            self.show_proceed_to_testing_button()
    


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

    def show_proceed_to_testing_button(self):
        # Hide the Stop Training button
        self.stop_button.pack_forget()
        # Create a button to proceed to testing
        self.proceed_button = tk.Button(self.master, text="Proceed to Testing", font=("Helvetica", 12, "bold"), fg="white", bg="green", command=self.transition_to_testing_gui)
        self.proceed_button.pack(pady=20)  # Adjust padding as necessary
    
    def stop_training(self):
        print("Stop training button clicked")

        # Directly stop the timer
        self.timer_running = False

        # Send stop request and log
        self.training_task_manager.stop_task()
        print("Stop request sent to training task manager")
        
        # Immediate GUI update to reflect stopping state
        self.status_label.config(text="Stopping Training...", fg="red")
        self.master.update_idletasks()  # Force immediate GUI update

        # Hide stop button and show stopping status
        self.stop_button.pack_forget()

        # After a short delay, complete the task
        self.master.after(1000, self.task_completed)

    def stop_training_process(self):
        print("Stop training process initiated")
        
        # Stop the current training task
        self.stop_training()  # Call the existing method to stop the current task
        
        # Set a flag or condition to prevent further tasks from starting
        self.training_process_stopped = True
        
        # Update the GUI to show that the training process is stopping
        self.status_label.config(text="Stopping Training Process...", fg="red")
        self.master.update_idletasks()  # Force immediate GUI update

         # Wait for a short moment to ensure that the current task is stopped
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=5)  # Wait for a maximum of 5 seconds for the thread to stop
        
        # After stopping, ensure that no further tasks are processed
        self.master.after(1000, self.finish_stopping_process)

    def finish_stopping_process(self):
        # Check if there are further tasks queued
        if self.current_task_index < len(self.task_list) - 1:
            print("Further tasks are queued, but training process is stopped.")
            self.status_label.config(text="Training process has been stopped. No further tasks will be processed.", fg="red")
        else:
            print("Training process stopped successfully.")
            self.status_label.config(text="Training process stopped successfully.", fg="red")

        # Optional: Provide an option to proceed to testing or close the GUI
        self.show_proceed_to_testing_button()


    def transition_to_testing_gui(self):
        # Destroy the current window and move to the testing GUI
        self.master.destroy()
        root = tk.Tk()
        VEstimTestingGUI(root)  # Assuming VEstimTestingGui is the class for the testing GUI
        root.mainloop()
    
    def on_closing(self):
        # Handle the window close event
        if self.training_thread.is_alive():
            self.stop_training()  # Stop the training thread
        self.master.destroy()  # Close the window

if __name__ == "__main__":
    root = tk.Tk()
    # Replace `task_list`, `params`, and `job_manager` with actual instances
    gui = VEstimTrainingTaskGUI(root, task_list, params, job_manager)
    root.mainloop()
