import tkinter as tk
from tkinter import filedialog, messagebox
import os
import time
import requests

class DataImportGUI:
    def __init__(self, master):
        self.master = master
        self.params = {}
        self.train_folder_path = ""
        self.test_folder_path = ""
        self.selected_files = {'train': [], 'test': []}

        self.setup_gui()

    def setup_gui(self):
        self.master.title("VEstim Modelling Tool")
        
        resources_path = os.path.join(os.path.dirname(__file__), 'resources')

        # Set window size and properties
        self.master.geometry("800x600")
        self.master.minsize(800, 600)
        self.master.state('zoomed')  # Start maximized

        icon_path = os.path.join(resources_path, 'icon.ico')
        self.master.iconbitmap(icon_path)
        self.master.configure(bg="#e3e8e8")
        self.master.eval('tk::PlaceWindow . center')
        self.master.attributes('-alpha', 0.95)
        
        self.label = tk.Label(self.master, text="Select data folders to train your LSTM Model", font=("Helvetica", 16, "bold"), fg="green")
        self.label.pack(pady=10)

        # Initialize the PanedWindow before defining the train and test listboxes
        paned_window = tk.PanedWindow(self.master, orient=tk.VERTICAL, sashrelief=tk.RAISED)
        paned_window.pack(fill=tk.BOTH, expand=True, pady=10)

        # Train Folder Selection and Listbox
        train_listbox_frame = tk.Frame(paned_window)
        self.select_train_folder_button = tk.Button(train_listbox_frame, text="Select Training Folder", command=self.select_train_folder, bg="#e6e6e6", relief=tk.GROOVE)
        self.select_train_folder_button.pack(side=tk.TOP, pady=5)

        train_scrollbar = tk.Scrollbar(train_listbox_frame, orient=tk.VERTICAL)
        self.train_files_listbox = tk.Listbox(train_listbox_frame, selectmode=tk.MULTIPLE, exportselection=False, yscrollcommand=train_scrollbar.set)
        train_scrollbar.config(command=self.train_files_listbox.yview)

        train_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.train_files_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Test Folder Selection and Listbox
        test_listbox_frame = tk.Frame(paned_window)
        self.select_test_folder_button = tk.Button(test_listbox_frame, text="Select Testing Folder", command=self.select_test_folder, bg="#e6e6e6", relief=tk.GROOVE)
        self.select_test_folder_button.pack(side=tk.TOP, pady=5)

        test_scrollbar = tk.Scrollbar(test_listbox_frame, orient=tk.VERTICAL)
        self.test_files_listbox = tk.Listbox(test_listbox_frame, selectmode=tk.MULTIPLE, exportselection=False, yscrollcommand=test_scrollbar.set)
        test_scrollbar.config(command=self.test_files_listbox.yview)

        test_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.test_files_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add frames to the PanedWindow
        paned_window.add(train_listbox_frame)
        paned_window.add(test_listbox_frame)


        # Button to organize files
        self.organize_button = tk.Button(self.master, text="Load and Prepare Files", command=self.organize_files, bg="#e6e6e6", relief=tk.GROOVE)
        self.organize_button.pack(pady=10)
        self.organize_button.config(state=tk.DISABLED)  # Initially disable the organize button

    def select_train_folder(self):
        self.train_folder_path = filedialog.askdirectory()
        if self.train_folder_path:
            print(f"Train folder selected: {self.train_folder_path}")  # Debugging statement
            self.populate_files_list(self.train_folder_path, is_train=True)
        self.check_folders_selected()

    def select_test_folder(self):
        self.test_folder_path = filedialog.askdirectory()
        if self.test_folder_path:
            print(f"Test folder selected: {self.test_folder_path}")  # Debugging statement
            self.populate_files_list(self.test_folder_path, is_train=False)
        self.check_folders_selected()

    def check_folders_selected(self):
        if self.train_folder_path and self.test_folder_path:
            self.organize_button.config(state=tk.NORMAL)
        else:
            self.organize_button.config(state=tk.DISABLED)

    def populate_files_list(self, folder_path, is_train=True):
        if is_train:
            self.train_files_listbox.delete(0, tk.END)
        else:
            self.test_files_listbox.delete(0, tk.END)
        
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".mat"):
                    file_path = os.path.join(root, file)
                    if is_train:
                        self.train_files_listbox.insert(tk.END, file_path)
                    else:
                        self.test_files_listbox.insert(tk.END, file_path)

    def organize_files(self):
        print("Starting to organize files...")  # Log message
        
        # Generate a unique job ID using a timestamp
        job_id = time.strftime("%Y%m%d-%H%M%S")
        print(f"Generated job ID: {job_id}")

        # Retrieve selected files from the listboxes
        train_files = [self.train_files_listbox.get(i) for i in self.train_files_listbox.curselection()]
        test_files = [self.test_files_listbox.get(i) for i in self.test_files_listbox.curselection()]
        
        print(f"Selected train files: {train_files}")
        print(f"Selected test files: {test_files}")

        # Check if both train and test files are selected
        if not train_files or not test_files:
            messagebox.showerror("Error", "No files selected for either training or testing.")
            return

        # Call data import microservice with file paths
        try:
            response = requests.post(
                "http://127.0.0.1:5001/upload",
                json={'train_files': train_files, 'test_files': test_files, 'job_id': job_id}
            )
            if response.status_code == 200:
                job_folder = response.json().get("job_folder")
                print(f"Files have been processed. Job folder: {job_folder}")
                messagebox.showinfo("Success", f"Files have been processed. Job folder: {job_folder}")
            else:
                print(f"Failed to process files: {response.status_code} - {response.text}")
                messagebox.showerror("Error", "Failed to process files. Please check the server logs for more details.")
        except Exception as e:
            print(f"Error during file organization: {e}")
            messagebox.showerror("Error", f"An error occurred during file organization: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    gui = DataImportGUI(root)
    root.mainloop()
