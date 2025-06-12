import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QListWidget, QFileDialog, QProgressBar, QComboBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
import os
import requests
import json
from datetime import datetime
import logging

from vestim.backend.src.services.data_processor.src.data_processor_qt_arbin import DataProcessorArbin
from vestim.backend.src.services.data_processor.src.data_processor_qt_stla import DataProcessorSTLA
from vestim.backend.src.services.data_processor.src.data_processor_qt_digatron import DataProcessorDigatron
from vestim.logger_config import setup_logger

logger = setup_logger(log_file='default.log')

DEFAULT_DATA_EXTENSIONS = [".csv", ".txt", ".mat", ".xls", ".xlsx", ".RES"]
SERVER_URL = "http://127.0.0.1:8001"

class DataImportWidget(QWidget):
    job_created = pyqtSignal(str)  # Emits job_id
    job_processing_complete = pyqtSignal(str) # Emits job_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.train_folder_path = ""
        self.test_folder_path = ""
        self.job_id = None
        self.data_processor_arbin = DataProcessorArbin()
        self.data_processor_stla = DataProcessorSTLA()
        self.data_processor_digatron = DataProcessorDigatron()
        self.organizer_thread = None
        self.organizer = None
        self.initUI()

    def initUI(self):
        self.main_layout = QVBoxLayout(self)

        # Header
        self.header_label = QLabel("Select data folders to train your LSTM Model", self)
        self.header_label.setAlignment(Qt.AlignCenter)
        self.header_label.setStyleSheet("font-size: 18px; font-weight: bold; color: green;")
        self.main_layout.addWidget(self.header_label)

        # Training folder section
        train_layout = QVBoxLayout()
        self.train_select_button = QPushButton("Select Train Data Folder", self)
        self.train_select_button.clicked.connect(self.select_train_folder)
        train_layout.addWidget(self.train_select_button)
        self.train_list_widget = QListWidget(self)
        self.train_list_widget.setSelectionMode(QListWidget.MultiSelection)
        train_layout.addWidget(self.train_list_widget)
        self.main_layout.addLayout(train_layout)

        # Testing folder section
        test_layout = QVBoxLayout()
        self.test_select_button = QPushButton("Select Test Data Folder", self)
        self.test_select_button.clicked.connect(self.select_test_folder)
        test_layout.addWidget(self.test_select_button)
        self.test_list_widget = QListWidget(self)
        self.test_list_widget.setSelectionMode(QListWidget.MultiSelection)
        test_layout.addWidget(self.test_list_widget)
        self.main_layout.addLayout(test_layout)

        # Data source and organize button
        combined_layout = QHBoxLayout()
        data_source_label = QLabel("Data Source:")
        combined_layout.addWidget(data_source_label)
        self.data_source_combo = QComboBox(self)
        self.data_source_combo.addItems(["Arbin", "STLA", "Digatron", "Biologic"])
        self.data_source_combo.currentIndexChanged.connect(self.on_data_source_selection_changed)
        combined_layout.addWidget(self.data_source_combo)
        combined_layout.addStretch(1)
        self.organize_button = QPushButton("Load and Prepare Files", self)
        self.organize_button.clicked.connect(self.organize_files)
        combined_layout.addWidget(self.organize_button)
        self.main_layout.addLayout(combined_layout)

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.main_layout.addWidget(self.progress_bar)

    def on_data_source_selection_changed(self, index):
        selected_source = self.data_source_combo.currentText()
        if self.train_folder_path:
            self.populate_file_list(self.train_folder_path, self.train_list_widget, selected_source)
        if self.test_folder_path:
            self.populate_file_list(self.test_folder_path, self.test_list_widget, selected_source)

    def select_train_folder(self):
        self.train_folder_path = QFileDialog.getExistingDirectory(self, "Select Training Folder")
        if self.train_folder_path:
            selected_source = self.data_source_combo.currentText()
            self.populate_file_list(self.train_folder_path, self.train_list_widget, selected_source)

    def select_test_folder(self):
        self.test_folder_path = QFileDialog.getExistingDirectory(self, "Select Testing Folder")
        if self.test_folder_path:
            selected_source = self.data_source_combo.currentText()
            self.populate_file_list(self.test_folder_path, self.test_list_widget, selected_source)

    def populate_file_list(self, folder_path, list_widget, data_source):
        list_widget.clear()
        if not folder_path or not os.path.isdir(folder_path):
            return
        
        if data_source == "Arbin":
            extensions_to_check = [".mat"]
        elif data_source == "Digatron":
            extensions_to_check = [".csv"]
        elif data_source == "STLA":
            extensions_to_check = [".xlsx", ".xls"]
        else:
            extensions_to_check = [ext.lower() for ext in DEFAULT_DATA_EXTENSIONS]
        
        for root, _, files in os.walk(folder_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions_to_check):
                    list_widget.addItem(os.path.join(root, file))

    def organize_files(self):
        train_files = [item.text() for item in self.train_list_widget.selectedItems()]
        test_files = [item.text() for item in self.test_list_widget.selectedItems()]

        if not train_files or not test_files:
            return

        self.organize_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        selected_source = self.data_source_combo.currentText()
        if selected_source == "Arbin":
            data_processor = self.data_processor_arbin
        elif selected_source == "STLA":
            data_processor = self.data_processor_stla
        elif selected_source == "Digatron":
            data_processor = self.data_processor_digatron
        else:
            return

        self.organizer = FileOrganizer(train_files, test_files, data_processor, data_source=selected_source)
        self.organizer_thread = QThread()
        self.organizer.progress.connect(self.progress_bar.setValue)
        self.organizer.job_id_signal.connect(self.on_job_created)
        self.organizer.job_folder_signal.connect(self.on_files_processed)
        self.organizer.moveToThread(self.organizer_thread)
        self.organizer_thread.started.connect(self.organizer.run)
        self.organizer_thread.start()

    def on_job_created(self, job_id):
        self.job_id = job_id
        self.job_created.emit(job_id)

    def on_files_processed(self, job_folder):
        self.progress_bar.setVisible(False)
        self.organize_button.setEnabled(True)
        self.job_processing_complete.emit(self.job_id)


class FileOrganizer(QObject):
    progress = pyqtSignal(int)
    job_folder_signal = pyqtSignal(str)
    job_id_signal = pyqtSignal(str)

    def __init__(self, train_files, test_files, data_processor, data_source=None):
        super().__init__()
        self.train_files = train_files
        self.test_files = test_files
        self.data_processor = data_processor
        self.data_source = data_source

    def run(self):
        if not self.train_files or not self.test_files:
            self.progress.emit(0)
            return

        try:
            job_id = self.create_job_via_api()
            if not job_id:
                self.progress.emit(0)
                return
            
            self.job_id_signal.emit(job_id)

            job_folder = self.data_processor.organize_and_convert_files(
                self.train_files,
                self.test_files,
                progress_callback=self.progress.emit,
                job_id=job_id
            )
            
            self.job_folder_signal.emit(job_folder)
        except Exception as e:
            self.progress.emit(0)
            
    def create_job_via_api(self):
        try:
            selections = {
                "train_files": [os.path.basename(f) for f in self.train_files],
                "test_files": [os.path.basename(f) for f in self.test_files],
                "train_folder": os.path.dirname(self.train_files[0]) if self.train_files else "",
                "test_folder": os.path.dirname(self.test_files[0]) if self.test_files else "",
                "data_source": self.data_source,
                "timestamp": datetime.now().isoformat()
            }
            
            response = requests.post(f"{SERVER_URL}/jobs", json={"selections": selections})
            response.raise_for_status()
            result = response.json()
            return result['job_id']
        except Exception as e:
            return None

if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = DataImportWidget()
    widget.show()
    sys.exit(app.exec_())