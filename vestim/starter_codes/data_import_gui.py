import tkinter as tk
from tkinter import filedialog, messagebox
import os
import shutil
import time
import scipy.io
import numpy as np

class DataImportGUI:
    def __init__(self, master):
        self.master = master
        self.params = {}
        self.train_folder_path = ""
        self.test_folder_path = ""

        self.setup_gui()

    def setup_gui(self):
        self.master.title("VEstim Modelling Tool")
        
        resources_path = os.path.join(os.path.dirname(__file__), '..', 'resources')

        # Set window size and properties
        self.master.geometry("600x400")
        self.master.minsize(600, 400)
        self.master.maxsize(1200, 800)
        
        icon_path = os.path.join(resources_path, 'icon.ico')
        self.master.iconbitmap(icon_path)
        self.master.configure(bg="#e3e8e8")
        self.master.eval('tk::PlaceWindow . center')
        self.master.attributes('-alpha', 0.95)
        
        self.label = tk.Label(self.master, text="Select data folders to train your LSTM Model")
        self.label.pack(pady=10)

        # Train Folder Selection
        self.select_train_folder_button = tk.Button(self.master, text="Select Training Folder", command=self.select_train_folder)
        self.select_train_folder_button.pack(pady=10)

        # Test Folder Selection
        self.select_test_folder_button = tk.Button(self.master, text="Select Testing Folder", command=self.select_test_folder)
        self.select_test_folder_button.pack(pady=10)

        # Listbox to display selected files
        self.files_listbox = tk.Listbox(self.master, selectmode=tk.MULTIPLE)
        self.files_listbox.pack(pady=10, fill=tk.BOTH, expand=True)

        # Button to organize files
        self.organize_button = tk.Button(self.master, text="Load and Prepare Files", command=self.organize_files)
        self.organize_button.pack(pady=10)
        self.organize_button.config(state=tk.DISABLED)  # Initially disable the organize button

    def select_train_folder(self):
        self.train_folder_path = filedialog.askdirectory()
        if self.train_folder_path:
            self.populate_files_list(self.train_folder_path)
        self.check_folders_selected()

    def select_test_folder(self):
        self.test_folder_path = filedialog.askdirectory()
        if self.test_folder_path:
            self.populate_files_list(self.test_folder_path)
        self.check_folders_selected()

    def check_folders_selected(self):
        if self.train_folder_path and self.test_folder_path:
            self.organize_button.config(state=tk.NORMAL)
        else:
            self.organize_button.config(state=tk.DISABLED)

    def populate_files_list(self, folder_path):
        self.files_listbox.delete(0, tk.END)
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".mat"):
                    self.files_listbox.insert(tk.END, os.path.join(root, file))

    def convert_mat_to_csv(self, mat_file, output_folder):
        data = scipy.io.loadmat(mat_file)
        if 'meas' in data:
            meas = data['meas'][0, 0]
            Timestamp = meas['Time'].flatten()
            Voltage = meas['Voltage'].flatten()
            Current = meas['Current'].flatten()
            Temp = meas['Battery_Temp_degC'].flatten()
            SOC = meas['SOC'].flatten()

            combined_data = np.column_stack((Timestamp, Voltage, Current, Temp, SOC))
            header = ['Timestamp', 'Voltage', 'Current', 'Temp', 'SOC']
            
            csv_file_name = os.path.join(output_folder, os.path.splitext(os.path.basename(mat_file))[0] + '.csv')
            np.savetxt(csv_file_name, combined_data, delimiter=",", header=",".join(header), comments='', fmt='%s')
            print(f'Data successfully written to {csv_file_name}')
        else:
            print(f'Skipping file {mat_file}: "meas" field not found')

    def organize_files(self):
        selected_files = [self.files_listbox.get(i) for i in self.files_listbox.curselection()]

        # Generate a unique job ID using a timestamp
        job_id = time.strftime("%Y%m%d-%H%M%S")

        # Create the job folder in a specific output directory
        output_dir = os.path.join(os.getcwd(), 'output')
        os.makedirs(output_dir, exist_ok=True)
        job_folder = os.path.join(output_dir, f"Job_{job_id}")
        os.makedirs(job_folder, exist_ok=True)

        # Organize selected files into timestamped subfolders inside the job folder
        if self.train_folder_path:
            train_subfolder = os.path.join(job_folder, f"train_{job_id}")
            os.makedirs(train_subfolder, exist_ok=True)
            self.organize_selected_files(selected_files, train_subfolder, "train")

        if self.test_folder_path:
            test_subfolder = os.path.join(job_folder, f"test_{job_id}")
            os.makedirs(test_subfolder, exist_ok=True)
            self.organize_selected_files(selected_files, test_subfolder, "test")

        messagebox.showinfo("Success", f"Files have been organized into the job folder: {job_folder}.")
        
        # Clear the window content for the next step
        for widget in self.master.winfo_children():
            widget.destroy()

        from src.gui.hyper_param_sel_gui import VEstimHyperParamGUI #import parameter_selection and go to next step
        VEstimHyperParamGUI(self.master, self.params)

    def organize_selected_files(self, selected_files, subfolder, folder_type):
        raw_folder = os.path.join(subfolder, "raw_data")
        processed_folder = os.path.join(subfolder, "processed_data")

        os.makedirs(raw_folder, exist_ok=True)
        os.makedirs(processed_folder, exist_ok=True)

        for file in selected_files:
            destination = os.path.join(raw_folder, os.path.basename(file))
            shutil.copy(file, destination)
            self.convert_mat_to_csv(destination, processed_folder)

        # Store the folder paths in the params dictionary for later use
        if folder_type == "train":
            self.params['TRAIN_FOLDER'] = processed_folder
        elif folder_type == "test":
            self.params['TEST_FOLDER'] = processed_folder

if __name__ == "__main__":
    root = tk.Tk()
    gui = DataImportGUI(root)
    root.mainloop()