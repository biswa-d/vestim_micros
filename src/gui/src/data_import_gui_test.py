import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import queue
import os, time
from src.gateway.src.job_manager import JobManager
from src.gui.src.hyper_param_gui_test import VEstimHyperParamGUI
from src.services.data_processor.src.data_processor_1_test import DataProcessor  # Import the combined DataProcessor

job_manager = JobManager()

class DataImportGUI:
    def __init__(self, master):
        self.master = master
        self.params = {}
        self.train_folder_path = ""
        self.test_folder_path = ""
        self.selected_files = {'train': [], 'test': []}
        self.job_manager = JobManager()  # Create an instance of JobManager
        self.data_processor = DataProcessor()  # Create an instance of DataProcessor
        self.queue = queue.Queue()

        self.setup_gui()

    def setup_gui(self):
        self.master.title("VEstim Modelling Tool")
        self.master.geometry("800x600")
        self.master.minsize(800, 600)
        self.master.configure(bg="#e3e8e8")
        self.master.attributes('-alpha', 0.99)

        self.label = tk.Label(self.master, text="Select data folders to train your LSTM Model", font=("Helvetica", 16, "bold"), fg="green")
        self.label.pack(pady=10)

        paned_window = tk.PanedWindow(self.master, orient=tk.VERTICAL, sashrelief=tk.RAISED)
        paned_window.pack(fill=tk.BOTH, expand=True, pady=10)

        train_listbox_frame = tk.Frame(paned_window)
        self.select_train_folder_button = tk.Button(train_listbox_frame, text="Select Training Folder", command=self.select_train_folder, bg="#e6e6e6", relief=tk.GROOVE)
        self.select_train_folder_button.pack(side=tk.TOP, pady=5)

        train_scrollbar = tk.Scrollbar(train_listbox_frame, orient=tk.VERTICAL)
        self.train_files_listbox = tk.Listbox(train_listbox_frame, selectmode=tk.MULTIPLE, exportselection=False, yscrollcommand=train_scrollbar.set)
        train_scrollbar.config(command=self.train_files_listbox.yview)
        train_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.train_files_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        test_listbox_frame = tk.Frame(paned_window)
        self.select_test_folder_button = tk.Button(test_listbox_frame, text="Select Testing Folder", command=self.select_test_folder, bg="#e6e6e6", relief=tk.GROOVE)
        self.select_test_folder_button.pack(side=tk.TOP, pady=5)

        test_scrollbar = tk.Scrollbar(test_listbox_frame, orient=tk.VERTICAL)
        self.test_files_listbox = tk.Listbox(test_listbox_frame, selectmode=tk.MULTIPLE, exportselection=False, yscrollcommand=test_scrollbar.set)
        test_scrollbar.config(command=self.test_files_listbox.yview)
        test_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.test_files_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        paned_window.add(train_listbox_frame)
        paned_window.add(test_listbox_frame)

        # Define the button and label in the setup
        self.organize_button = tk.Button(
            self.master,
            text="Load and Prepare Files",
            command=self.organize_files,
            bg="#ccffcc",  
            fg="black",  
            activebackground="#b3ffb3",  
            relief=tk.GROOVE
        )
        self.organize_button.pack(pady=10)

        # Updated progress label with light blue background
        self.progress_label = tk.Label(
            self.master,
            text="Processing...",
            font=("Helvetica", 10),
            bg="#e6f7ff",  # Light blue background
            fg="black"
        )
        self.progress_label.pack(pady=10)
        self.progress_label.pack_forget()  # Hide initially

    # Animate loading function
    def animate_loading(self):
        loading_text = "Processing"
        num_dots = (int(time.time() * 2) % 4)  # Change number of dots based on time
        self.progress_label.config(text=loading_text + "." * num_dots)
        self.master.after(500, self.animate_loading)  # Update every 500ms

    def select_train_folder(self):
        self.train_folder_path = filedialog.askdirectory()
        if self.train_folder_path:
            self.populate_files_list(self.train_folder_path, is_train=True)
        self.check_folders_selected()

    def select_test_folder(self):
        self.test_folder_path = filedialog.askdirectory()
        if self.test_folder_path:
            self.populate_files_list(self.test_folder_path, is_train=False)
        self.check_folders_selected()

    def check_folders_selected(self):
        if self.train_folder_path and self.test_folder_path:
            self.organize_button.config(state=tk.NORMAL)
        else:
            self.organize_button.config(state=tk.DISABLED)

    def populate_files_list(self, folder_path, is_train=True):
        listbox = self.train_files_listbox if is_train else self.test_files_listbox
        listbox.delete(0, tk.END)
        
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".mat"):
                    file_path = os.path.join(root, file)
                    listbox.insert(tk.END, file_path)

    def organize_files(self):
        self.progress_label.pack()  # Show the processing label
        threading.Thread(target=self._background_organize_files).start()
        self.process_queue()  # Start processing the queue to handle background updates

    def _background_organize_files(self):
        train_files = [self.train_files_listbox.get(i) for i in self.train_files_listbox.curselection()]
        test_files = [self.test_files_listbox.get(i) for i in self.test_files_listbox.curselection()]

        if not train_files or not test_files:
            self.queue.put("error: No files selected for either training or testing.")
            return

        try:
            job_folder = self.data_processor.organize_and_convert_files(train_files, test_files)
            self.queue.put(f"success: Files have been processed. Job folder: {job_folder}")
        except Exception as e:
            self.queue.put(f"error: An error occurred during file organization: {e}")

    def process_queue(self):
        try:
            message = self.queue.get_nowait()
            if message.startswith("success"):
                self.progress_label.config(text="Completed!")
                self.move_to_next_screen()  # Transition to the next GUI screen
            elif message.startswith("error"):
                messagebox.showerror("Error", message.split("error: ")[1])
            self.progress_label.pack_forget()  # Hide the processing label when done
        except queue.Empty:
            self.master.after(100, self.process_queue)

    def move_to_next_screen(self):
        for widget in self.master.winfo_children():
            widget.destroy()

        # params = {}  # Pass any initial parameters if needed
        gui = VEstimHyperParamGUI(self.master)
                
if __name__ == "__main__":
    root = tk.Tk()
    gui = DataImportGUI(root)
    root.mainloop()

