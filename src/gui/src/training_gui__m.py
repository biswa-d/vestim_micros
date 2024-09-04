import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from queue import Queue, Empty
from threading import Thread
import time
from gateway.src.testing_manager import VEstimTrainingManager
from src.gateway.src.job_manager import JobManager

class VEstimTrainingGUI:
    def __init__(self, master, params, job_manager):
        self.master = master
        self.params = params
        self.job_manager = job_manager

        # Initialize variables
        self.train_loss_values = []
        self.valid_loss_values = []
        self.valid_x_values = []
        self.start_time = None
        self.training_thread = None
        self.queue = Queue()
        self.timer_running = True

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
        "REPETITIONS": "Repetitions"}

        # Initialize the Training Manager
        self.training_manager = VEstimTrainingManager(self.params, self.queue, self.job_manager)

        self.build_gui()

    def build_gui(self):
        # Clear the existing content from the master window
        for widget in self.master.winfo_children():
            widget.destroy()

        # Update the window title and show initial message
        self.master.title("VEstim - Building LSTM Model")
        self.master.geometry("900x600")
        self.master.minsize(900, 600)

        # Create main frame
        main_frame = tk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        main_frame.pack_propagate(False)

        # Title Label
        title_label = tk.Label(main_frame, text="Training LSTM Model with Hyperparameters", font=("Helvetica", 12, "bold"))
        title_label.pack(pady=5)

        # Frame for hyperparameter labels
        frame = tk.Frame(main_frame)
        frame.pack(fill=tk.X, pady=5)

        # Filter out the TRAIN_FOLDER and TEST_FOLDER from being displayed
        params_items = [(key, value) for key, value in self.params.items() if key not in ['TRAIN_FOLDER', 'TEST_FOLDER']]
        columns = [params_items[i::5] for i in range(5)]  # Splits into five columns

        # Display each column with labels
        for col_num, column in enumerate(columns):
            col_frame = tk.Frame(frame)
            col_frame.grid(row=0, column=col_num, padx=5)
            for row_num, (param, value) in enumerate(column):
                label_text = self.param_labels.get(param, param)  # Get the user-friendly label or fallback to the key
                param_label = tk.Label(col_frame, text=f"{label_text}: ", font=("Helvetica", 10))  
                value_label = tk.Label(col_frame, text=f"{value}", font=("Helvetica", 10, "bold"))  
                param_label.grid(row=row_num, column=0, sticky='w')
                value_label.grid(row=row_num, column=1, sticky='w')

        # Centering the hyperparameters table
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(len(columns) - 1, weight=1)

        # Initialize status label
        self.status_label = tk.Label(main_frame, text="Training the LSTM model...", fg="green", font=("Helvetica", 10, "bold"))
        self.status_label.pack(pady=5)

        # Create a frame to hold both labels
        time_frame = tk.Frame(main_frame)
        time_frame.pack(pady=5)

        # Label for static text
        self.static_text_label = tk.Label(time_frame, text="Time Since Training Started:", fg="blue", font=("Helvetica", 10))
        self.static_text_label.pack(side=tk.LEFT)

        # Label for the dynamic time value in bold
        self.time_value_label = tk.Label(time_frame, text="00h:00m:00s", fg="blue", font=("Helvetica", 10, "bold"))
        self.time_value_label.pack(side=tk.LEFT)

        # Setting up Matplotlib figure for loss plots with adjusted size
        fig = Figure(figsize=(6, 2.5), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.set_xlabel("Epoch", labelpad=0)
        self.ax.set_ylabel("Loss [% RMSE]")
        self.ax.set_xlim(1, self.params['MAX_EPOCHS'])
        xticks = list(range(1, self.params['MAX_EPOCHS'] + 1, self.params['ValidFrequency']))
        if self.params['MAX_EPOCHS'] not in xticks:
            xticks.append(self.params['MAX_EPOCHS'])  
        self.ax.set_xticks(xticks)
        self.ax.set_title("Training and Validation Loss")
        self.ax.legend(["Train Loss", "Validation Loss"])
        self.ax.plot([], [], label='Train Loss')
        self.ax.plot([], [], label='Validation Loss')
        self.ax.legend()  
        self.canvas = FigureCanvasTkAgg(fig, master=main_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.ax.margins(y=0.1)
        fig.subplots_adjust(bottom=0.2)

        # Rolling window for displaying detailed logs
        self.log_text = tk.Text(main_frame, height=1, wrap=tk.WORD)
        self.log_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.log_text.insert(tk.END, "Initial Parameters:\n")
        self.log_text.insert(tk.END, f"Repetition: 1/{self.params['REPETITIONS']}, Epoch: 0, Train Loss: N/A, Validation Error: N/A\n")
        self.log_text.see(tk.END)

        # Stop button
        self.stop_button = tk.Button(main_frame, text="Stop Training", command=self.stop_training, bg="red", fg="white")
        self.stop_button.pack(pady=10)

        # Start training
        self.start_training()

    def update_progress(self, repetition, epoch, train_loss, validation_error):
        if epoch == 1 or epoch % self.params['ValidFrequency'] == 0 or epoch == self.params['MAX_EPOCHS']:
            self.valid_x_values.append(epoch)
            self.train_loss_values.append(train_loss[-1] * 100)  
            self.valid_loss_values.append(validation_error[-1] * 100)  

            self.ax.clear()
            self.ax.plot(self.valid_x_values, self.train_loss_values, label='Train Loss')
            self.ax.plot(self.valid_x_values, self.valid_loss_values, label='Validation Loss')
            self.ax.set_xlim(1, self.params['MAX_EPOCHS'])
            xticks = list(range(1, self.params['MAX_EPOCHS'] + 1, self.params['ValidFrequency']))
            if self.params['MAX_EPOCHS'] not in xticks:
                xticks.append(self.params['MAX_EPOCHS'])
            self.ax.set_xticks(xticks)
            self.ax.legend()
            self.ax.set_xlabel("Epoch", labelpad=15)
            self.ax.set_ylabel("Loss [% RMSE]")
            self.ax.set_title("Training and Validation Loss")
            self.ax.xaxis.set_label_coords(0.5, -0.1)
            self.canvas.draw()

        self.log_text.insert(tk.END, f"Repetition: {repetition}, Epoch: {epoch}, Train Loss: {train_loss[-1] * 100:.4f}%, Validation Loss: {validation_error[-1] * 100:.4f}%\n")
        self.log_text.see(tk.END)

    def update_elapsed_time(self):
        if self.timer_running:
            elapsed_time = time.time() - self.start_time
            self.final_time = elapsed_time  
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.time_value_label.config(text=f" {int(hours):02}h:{int(minutes):02}m:{int(seconds):02}s")
            self.master.after(1000, self.update_elapsed_time)

    def start_training(self):
        self.start_time = time.time()

        def run_training():
            self.training_manager.start_training(self.update_progress)
            self.master.after(0, lambda: self.show_results())

        self.training_thread = Thread(target=run_training)
        self.training_thread.setDaemon(True)
        self.training_thread.start()

        self.update_elapsed_time()
        self.process_queue()

    def process_queue(self):
        try:
            while True:
                repetition, epoch, train_loss, validation_error = self.queue.get_nowait()
                self.update_progress(repetition, epoch, train_loss, validation_error)
        except Empty:
            self.master.after(100, self.process_queue)

    def stop_training(self):
        if self.training_thread.is_alive():
            self.timer_running = False
            elapsed_time = time.time() - self.start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.time_value_label.config(text=f"Training Time: {int(hours):02}h:{int(minutes):02}m:{int(seconds):02}s")
            self.status_label.config(text="Training Stopped! Saving Model...", fg="red")
            
            # Call the stop_training method from the training manager
            self.training_manager.stop_training()

            self.stop_button.pack_forget()
            self.show_results()


    def show_results(self):
        self.timer_running = False  

        hours, remainder = divmod(self.final_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        formatted_time = f"{int(hours):02}h:{int(minutes):02}m:{int(seconds):02}s"

        self.static_text_label.config(text="Training Time Taken:")
        self.time_value_label.config(text=formatted_time)

        self.status_label.config(text="Training Completed")

        self.stop_button.destroy()
        proceed_button = tk.Button(self.master, text="Proceed to Testing", command=self.proceed_to_testing, bg="green", fg="white")
        proceed_button.pack(pady=10)

    def proceed_to_testing(self):
        pass

if __name__ == "__main__":
    root = tk.Tk()
    params = {}
    job_manager = JobManager()
    gui = VEstimTrainingGUI(root, params, job_manager)
    root.mainloop()
