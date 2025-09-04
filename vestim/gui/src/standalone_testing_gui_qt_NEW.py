import sys
import os
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QWidget, QTreeWidget, QTreeWidgetItem, QProgressBar, QFrame, QMessageBox,
                             QGroupBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

# Import the testing manager EXACTLY like main testing GUI
from vestim.gateway.src.job_manager_qt import JobManager
from vestim.gateway.src.testing_manager_qt import VEstimTestingManager


class VEstimStandaloneTestingGUI(QMainWindow):
    def __init__(self, job_folder_path):
        super().__init__()
        self.job_folder_path = job_folder_path
        
        # Initialize managers EXACTLY like main testing GUI
        self.job_manager = JobManager()
        self.job_manager.set_job_folder(job_folder_path)
        
        # Load hyperparameters from job folder
        self.load_hyperparams_from_job()
        
        # Initialize testing manager with the same parameters as main testing GUI
        self.testing_manager = VEstimTestingManager(
            job_manager=self.job_manager, 
            params=self.params, 
            task_list=None,  # Will be loaded from models directory
            training_results={}
        )
        
        self.results_list = []
        self.sl_no_counter = 1
        
        self.initUI()

    def load_hyperparams_from_job(self):
        """Load hyperparameters from job folder EXACTLY like main testing GUI expects"""
        hyperparams_file = os.path.join(self.job_folder_path, 'hyperparams.json')
        if os.path.exists(hyperparams_file):
            with open(hyperparams_file, 'r') as f:
                self.params = json.load(f)
        else:
            self.params = {}
            print(f"Warning: No hyperparams.json found in {self.job_folder_path}")

    def initUI(self):
        self.setWindowTitle(f"VEstim Tool - Standalone Testing Results: {os.path.basename(self.job_folder_path)}")
        self.setGeometry(100, 100, 1200, 800)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Apply the EXACT same styling as main testing GUI
        self.setStyleSheet("""
            QPushButton:disabled {
                background-color: #d3d3d3;
                color: #a9a9a9;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
        """)

        title_label = QLabel("Standalone Testing Results")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #0b6337; margin-bottom: 20px;")
        self.main_layout.addWidget(title_label)

        # Create hyperparameters frame EXACTLY like main testing GUI
        self.hyperparam_frame = QFrame()
        self.hyperparam_frame.setObjectName("hyperparamFrame")
        self.hyperparam_frame.setStyleSheet("""
            #hyperparamFrame {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                background-color: #ffffff;
            }
        """)
        self.main_layout.addWidget(self.hyperparam_frame)
        
        # Display hyperparameters using the EXACT same method as main testing GUI
        self.display_hyperparameters(self.params)

        # Create results table section EXACTLY like main testing GUI
        self.create_results_section()

    def display_hyperparameters(self, params):
        """Display hyperparameters using the EXACT same method as main testing GUI"""
        from vestim.gui.src.adaptive_gui_utils import display_hyperparameters
        display_hyperparameters(self, params)

    def create_results_section(self):
        """Create the results table section exactly like main testing GUI"""
        results_group = QGroupBox("Testing Results")
        results_layout = QVBoxLayout(results_group)
        
        # Create results table
        self.results_table = QTreeWidget()
        self.results_table.setHeaderLabels(["Model", "Task", "File", "MAE", "MSE", "RMSE", "MAPE", "R²", "Actions"])
        self.results_table.setRootIsDecorated(False)
        self.results_table.setAlternatingRowColors(True)
        
        # Set column widths
        header = self.results_table.header()
        header.resizeSection(0, 100)  # Model
        header.resizeSection(1, 100)  # Task  
        header.resizeSection(2, 150)  # File
        header.resizeSection(3, 80)   # MAE
        header.resizeSection(4, 80)   # MSE
        header.resizeSection(5, 80)   # RMSE
        header.resizeSection(6, 80)   # MAPE
        header.resizeSection(7, 80)   # R²
        header.resizeSection(8, 100)  # Actions
        
        results_layout.addWidget(self.results_table)
        self.main_layout.addWidget(results_group)
        
        # Progress bar at bottom
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(0)  # Indeterminate
        self.progress_bar.hide()  # Initially hidden
        self.main_layout.addWidget(self.progress_bar)

    def add_result_row(self, result):
        """Add result row EXACTLY like main testing GUI"""
        task_data_for_log = result.get('task_completed', {})
        log_summary = (
            f"Model: {task_data_for_log.get('model_name', 'N/A')}, "
            f"Task: {task_data_for_log.get('task_name', 'N/A')}, "
            f"File: {task_data_for_log.get('file_name', 'N/A')}"
        )
        print(f"[DEBUG] Adding result row: {log_summary}")

        if 'task_error' in result:
            print(f"Error in task: {result['task_error']}")
            return

        task_data = result.get('task_completed')

        if task_data:
            save_dir = task_data.get("saved_dir", "")
            model_name = task_data.get("model_name", "N/A")
            task_name = task_data.get("task_name", "N/A")
            task_id = task_data.get("task_info", {}).get("task_id", "")
            file_name = task_data.get("file_name", "Unknown File")
            
            target_column_name = task_data.get("target_column", "")
            predictions_file = task_data.get("predictions_file", "")

            # Get error metrics
            mae = task_data.get("mae", "N/A")
            mse = task_data.get("mse", "N/A")  
            rmse = task_data.get("rmse", "N/A")
            mape = task_data.get("mape", "N/A")
            r2 = task_data.get("r2", "N/A")

            # Create tree widget item
            item = QTreeWidgetItem(self.results_table)
            item.setText(0, model_name)
            item.setText(1, task_name)
            item.setText(2, file_name)
            item.setText(3, f"{mae:.4f}" if isinstance(mae, (int, float)) else str(mae))
            item.setText(4, f"{mse:.4f}" if isinstance(mse, (int, float)) else str(mse))
            item.setText(5, f"{rmse:.4f}" if isinstance(rmse, (int, float)) else str(rmse))
            item.setText(6, f"{mape:.2f}%" if isinstance(mape, (int, float)) else str(mape))
            item.setText(7, f"{r2:.4f}" if isinstance(r2, (int, float)) else str(r2))
            
            # Create plot button EXACTLY like main testing GUI
            plot_button = QPushButton("Plot")
            plot_button.setStyleSheet("""
                QPushButton {
                    background-color: #0b6337;
                    color: white;
                    font-weight: bold;
                    padding: 5px 15px;
                    border-radius: 3px;
                    border: none;
                }
                QPushButton:hover {
                    background-color: #0d7940;
                }
            """)
            plot_button.clicked.connect(lambda: self.plot_model_result(predictions_file, save_dir, target_column_name))
            self.results_table.setItemWidget(item, 8, plot_button)

    def plot_model_result(self, predictions_file, save_dir, target_column_name):
        """Plot model result EXACTLY like main testing GUI"""
        # This will use the exact same plotting logic as main testing GUI
        try:
            # For now, just show a message that plotting will be implemented
            QMessageBox.information(self, "Plot", f"Plot for {predictions_file} will be shown here (exact same as main testing GUI)")
            print(f"[DEBUG] Plot requested for: {predictions_file}")
        except Exception as e:
            QMessageBox.warning(self, "Plot Error", f"Error showing plot: {str(e)}")


def main():
    """Main function to run the standalone testing GUI"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("VEstim Standalone Testing")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("McMaster University Battery Lab")
    
    # For testing purposes, create with a dummy path
    gui = VEstimStandaloneTestingGUI("test_job_folder")
    gui.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()