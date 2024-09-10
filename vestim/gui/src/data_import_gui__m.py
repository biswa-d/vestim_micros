import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime
import requests
import os
from vestim.gateway.src.job_manager import JobManager  # Import the JobManager class
from vestim.gui.src.hyper_param_gui import VEstimHyperParamGUI  # Import the Hyperparameter GUI

job_manager = JobManager()

class DataImportGUI:
    def __init__(self, master):
        self.master = master
        self.params = {}
        self.train_folder_path = ""
        self.test_folder_path = ""
        self.selected_files = {'train': [], 'test': []}
        self.job_manager = JobManager()  # Create an instance of JobManager

        self.setup_gui()

    def setup_gui(self):
        self.master.title("VEstim Modelling Tool")
        self.master.geometry("800x600")
        self.master.minsize(800, 600)
        # self.master.state('zoomed')
        self.master.configure(bg="#e3e8e8")
        self.master.attributes('-alpha', 0.95)

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

        self.organize_button = tk.Button(
        self.master,
        text="Load and Prepare Files",
        command=self.organize_files,
        bg="#ccffcc",  # Light green background
        fg="black",  # Black text color for better contrast with light green
        activebackground="#b3ffb3",  # Even lighter green when the button is clicked
        relief=tk.GROOVE)
       

        self.organize_button.pack(pady=10)
        self.organize_button.config(state=tk.DISABLED)

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
        job_id, job_folder = self.job_manager.create_new_job()
        train_files = [self.train_files_listbox.get(i) for i in self.train_files_listbox.curselection()]
        test_files = [self.test_files_listbox.get(i) for i in self.test_files_listbox.curselection()]

        if not train_files or not test_files:
            messagebox.showerror("Error", "No files selected for either training or testing.")
            return

        try:
            response = requests.post(
                "http://127.0.0.1:5001/upload",
                json={'train_files': train_files, 'test_files': test_files, 'job_id': job_id}
            )
            if response.status_code == 200:
                print(f"Files have been processed. Job folder: {job_folder}")
                messagebox.showinfo("Success", f"Files have been processed. Job folder: {job_folder}")
                # Clear the current window
                for widget in self.master.winfo_children():
                    widget.destroy()

                # Initialize and display the Hyperparameter GUI
                params = {}  # Pass any initial parameters if needed
                gui = VEstimHyperParamGUI(self.master, params)
            else:
                print(f"Failed to process files: {response.status_code} - {response.text}")
                messagebox.showinfo("Success", f"Files have been processed. Job folder: {job_folder}\n\nMoving to Parameter Selection")
        except Exception as e:
            print(f"Error during file organization: {e}")
            messagebox.showerror("Error", f"An error occurred during file organization: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    gui = DataImportGUI(root)
    root.mainloop()
