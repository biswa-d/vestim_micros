import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from queue import Queue, Empty
from threading import Thread
import time
from src.gateway.src.training_manager import VEstimTrainingManager
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
            "REPETITIONS": "Repetitions"
        }

        # Initialize the Training Manager
        self.training_manager = VEstimTrainingManager(self.params, self.queue, self.job_manager)

        self.build_gui()

    def build_gui(self):
        # Clear the existing content from the master window
        for widget in self.master.winfo_children():
            widget.destroy()

        # Update the window title and show initial message
        self.master.title("VEstim - Training LSTM Model")
        self.master.geometry("900x600")
        self.master.minsize(900, 600)

        # Create main frame
        main_frame = tk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        main_frame.pack_propagate(False)

        # Title Label
        title_label = tk.Label(main_frame, text="Training LSTM models with hyperparameter set", font=("Helvetica", 12, "bold"))
        title_label.pack(pady=5)

        # Frame for hyperparameter labels
        self.hyperparam_frame = tk.Frame(main_frame)
        self.hyperparam_frame.pack(fill=tk.X, pady=5)

        # Display placeholders for hyperparameter values
        self.display_hyperparameters(self.params)

        # Initialize status label
        self.status_label = tk.Label(main_frame, text="Starting training...", fg="green", font=("Helvetica", 10, "bold"))
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

    def update_progress(self, progress_data):
        # Update the GUI with the current training progress
        pass  # Implementation as before

    def display_hyperparameters(self, params):
        # Display hyperparameters in the GUI
        pass  # Implementation as before

    def update_elapsed_time(self):
        # Update the elapsed time
        pass  # Implementation as before

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
                progress_data = self.queue.get_nowait()
                self.update_progress(progress_data)
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
