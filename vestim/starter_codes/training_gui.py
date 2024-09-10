#----------------------------------------------------------------------------------------
#Descrition: This file contains the final tested code for the training GUI state machine. It is responsible for training the LSTM model and updating the progress in the GUI.
#
# Created On: Wed Aug 07 2024 16:14:26
# Author: Biswanath Dehury
# Company: Dr. Phil Kollmeyer's Battery Lab at McMaster University
# Copyright (c) 2024 Biswanath Dehury, Dr. Phil Kollmeyer's Battery Lab at McMaster University
#----------------------------------------------------------------------------------------


import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from queue import Queue, Empty
from threading import Thread
from ..model_training.LSTM_Cell_model import build_lstm_model
from ..model_training.train_model import train
import time

class VEstimTrainingGUI:
    def __init__(self, master, params):
        self.master = master
        self.params = params

        # Initialize variables
        self.train_loss_values = []
        self.valid_loss_values = []
        self.valid_x_values = []
        self.start_time = None
        self.training_thread = None
        self.queue = Queue()
        self.timer_running = True

        self.build_gui()

    def build_gui(self):
        # Clear the existing content from the master window
        for widget in self.master.winfo_children():
            widget.destroy()

        # Update the window title and show initial message
        self.master.title("VEstim - Building LSTM Model")
        self.master.geometry("900x600")  # Smaller default size
        self.master.minsize(900, 600)

        # Create main frame
        main_frame = tk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        main_frame.pack_propagate(False)  # Prevent automatic resizing

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
                param_label = tk.Label(col_frame, text=f"{param}: ", font=("Helvetica", 10))  # Regular font for label
                value_label = tk.Label(col_frame, text=f"{value}", font=("Helvetica", 10, "bold"))  # Bold font for value

                # Use grid to ensure both labels stay on the same line
                param_label.grid(row=row_num, column=0, sticky='w')
                value_label.grid(row=row_num, column=1, sticky='w')

        # Centering the hyperparameters table
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(len(columns) - 1, weight=1)

        # Initialize status label
        self.status_label = tk.Label(main_frame, text="Training the LSTM model...", fg="green", font=("Helvetica", 10, "bold"))
        self.status_label.pack(pady=5)

        # # Initialize elapsed time label
        # self.elapsed_time_label = tk.Label(main_frame, text="Time Since Training Started: 00h:00m:00s", fg="blue", font=("Helvetica", 10))
        # self.elapsed_time_label.pack(pady=5) # commented out to make the time value label bold in the code below
        
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
        fig = Figure(figsize=(6, 2.5), dpi=100)  # Slightly larger plot size
        self.ax = fig.add_subplot(111)
        self.ax.set_xlabel("Epoch", labelpad=0)  # Increase padding for x-label
        self.ax.set_ylabel("Loss [% RMSE]")
        # self.ax.set_xlim(0, self.params['MAX_EPOCHS'])
        self.ax.set_xlim(1, self.params['MAX_EPOCHS'])  # Adjust x-axis limit for 1-based indexing
        # Set custom x-ticks to ensure the last epoch is included
        xticks = list(range(1, self.params['MAX_EPOCHS'] + 1, self.params['ValidFrequency']))
        if self.params['MAX_EPOCHS'] not in xticks:
            xticks.append(self.params['MAX_EPOCHS'])  # Ensure the last epoch is included
        self.ax.set_xticks(xticks)
        self.ax.set_title("Training and Validation Loss")
        self.ax.legend(["Train Loss", "Validation Loss"])
        # Initialize empty lines for the legend to appear
        self.ax.plot([], [], label='Train Loss')
        self.ax.plot([], [], label='Validation Loss')
        self.ax.legend()  # Display legend immediately
        self.canvas = FigureCanvasTkAgg(fig, master=main_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # Adjust plot margins to ensure x-label visibility
        self.ax.margins(y=0.1)  # Adjust y-margin to make room for the x-label
        fig.subplots_adjust(bottom=0.2)  # Increase the bottom margin to ensure x-label is visible

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
        # Append and plot if the current epoch is at the validation frequency
        if epoch == 1 or epoch % self.params['ValidFrequency'] == 0 or epoch == self.params['MAX_EPOCHS']:
            self.valid_x_values.append(epoch)
            self.train_loss_values.append(train_loss[-1] * 100)  # Convert to percentage
            self.valid_loss_values.append(validation_error[-1] * 100)  # Convert to percentage

            self.ax.clear()
            self.ax.plot(self.valid_x_values, self.train_loss_values, label='Train Loss')
            self.ax.plot(self.valid_x_values, self.valid_loss_values, label='Validation Loss')
            self.ax.set_xlim(1, self.params['MAX_EPOCHS'])  # Adjust x-axis limit for 1-based indexing
            # Set custom x-ticks to ensure the last epoch is included
            xticks = list(range(1, self.params['MAX_EPOCHS'] + 1, self.params['ValidFrequency']))
            if self.params['MAX_EPOCHS'] not in xticks:
                xticks.append(self.params['MAX_EPOCHS'])  # Ensure the last epoch is included
            self.ax.set_xticks(xticks)
            self.ax.legend()
            self.ax.set_xlabel("Epoch", labelpad=15)  # Ensure padding is maintained
            self.ax.set_ylabel("Loss [% RMSE]")  # Display percentage RMSE
            self.ax.set_title("Training and Validation Loss")
            self.ax.xaxis.set_label_coords(0.5, -0.1)  # Adjust the coordinates to keep the label in place
            self.canvas.draw()

        # Update logs in the scrollable window every validation epoch
        self.log_text.insert(tk.END, f"Repetition: {repetition}, Epoch: {epoch}, Train Loss: {train_loss[-1] * 100:.4f}%, Validation Loss: {validation_error[-1] * 100:.4f}%\n")
        self.log_text.see(tk.END)

    def update_elapsed_time(self):
        if self.timer_running:
            elapsed_time = time.time() - self.start_time
            self.final_time = elapsed_time  # Store the final elapsed time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.time_value_label.config(text=f" {int(hours):02}h:{int(minutes):02}m:{int(seconds):02}s")
            self.master.after(1000, self.update_elapsed_time)


    def start_training(self):
        self.start_time = time.time()

        def run_training():
            model = build_lstm_model(self.params)
            self.status_label.config(text="Training the LSTM model...", fg="green")
            self.master.update_idletasks()

            best_model = train(model, self.params, lambda *args: self.queue.put(args))
            self.master.after(0, lambda: self.show_results(best_model))

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
            # Stop the timer and update status
            self.timer_running = False
            elapsed_time = time.time() - self.start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.elapsed_time_label.config(text=f"Training Time: {int(hours):02}h:{int(minutes):02}m:{int(seconds):02}s")
            self.status_label.config(text="Training Stopped! Saving Model...", fg="red")

            # Save model here and stop training
            print("Training stopped. Saving model...")
            self.stop_button.pack_forget()
            self.show_results(None)  # Show results directly after stopping

    def show_results(self, best_model):
        # Update status and proceed button
        self.timer_running = False  # Ensure timer stops

        # Format the final elapsed time
        hours, remainder = divmod(self.final_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        formatted_time = f"{int(hours):02}h:{int(minutes):02}m:{int(seconds):02}s"

        # # Update the elapsed time label to show the final training time
        # self.elapsed_time_label.config(text=f"Training Time Taken: {formatted_time}") # commented out to make the time value label bold in the code below

        # Update the static text label to show "Training Time Taken:"
        self.static_text_label.config(text="Training Time Taken:")
        # Update the dynamic time label to show the final training time
        self.time_value_label.config(text=formatted_time)

        # Update the status label to indicate training completion
        self.status_label.config(text="Training Completed")

        # Replace the Stop Training button with the Proceed to Testing button
        self.stop_button.destroy()
        proceed_button = tk.Button(self.master, text="Proceed to Testing", command=lambda: self.proceed_to_testing(best_model), bg="green", fg="white")
        proceed_button.pack(pady=10)

    def proceed_to_testing(self, best_model):
        # Implement the transition to the testing GUI
        pass

# To use the class
# root = tk.Tk()
# app = VEstimTrainingGUI(root, params)
# root.mainloop()

