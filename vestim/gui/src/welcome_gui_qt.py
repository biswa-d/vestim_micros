import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QLabel, QWidget
from PyQt5.QtCore import Qt
from vestim.gui.src.data_import_gui_qt import DataImportGUI
from vestim.gui.src.test_selection_gui_qt import TestSelectionGUI
from vestim.gui.src.adaptive_gui_utils import get_adaptive_stylesheet, scale_widget_size

class WelcomeGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Welcome to PyBattML")
        self.initUI()

    def initUI(self):
        # Set window size consistent with DataImportGUI
        self.setGeometry(100, 100, scale_widget_size(1200), scale_widget_size(800))

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setAlignment(Qt.AlignCenter)
        self.main_layout.setSpacing(15) # Adjusted spacing
        self.main_layout.addStretch(1) # Add stretch to push content to the center vertically

        # Welcome Message
        welcome_label = QLabel("Welcome to PyBattML")
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setStyleSheet(get_adaptive_stylesheet("font-size: 32pt; font-weight: bold; color: #0b6337;"))
        self.main_layout.addWidget(welcome_label)

        description_label = QLabel("A Machine Learning Toolbox for Battery Datasets")
        description_label.setAlignment(Qt.AlignCenter)
        description_label.setStyleSheet(get_adaptive_stylesheet("font-size: 14pt; color: #555; margin-bottom: 30px;"))
        self.main_layout.addWidget(description_label)

        # --- Buttons ---
        self.new_training_button = QPushButton("Start New Training")
        self.new_training_button.clicked.connect(self.start_new_training)
        self.main_layout.addWidget(self.new_training_button, alignment=Qt.AlignCenter)

        training_desc = QLabel("Build, train, and evaluate a new model from scratch.")
        training_desc.setAlignment(Qt.AlignCenter)
        training_desc.setStyleSheet(get_adaptive_stylesheet("font-size: 11pt; color: #666;"))
        self.main_layout.addWidget(training_desc, alignment=Qt.AlignCenter)

        self.main_layout.addSpacing(40)

        self.test_model_button = QPushButton("Test a Trained Model")
        self.test_model_button.clicked.connect(self.test_trained_model)
        self.main_layout.addWidget(self.test_model_button, alignment=Qt.AlignCenter)

        test_desc = QLabel("Test trained models with comprehensive metrics and result plots.")
        test_desc.setAlignment(Qt.AlignCenter)
        test_desc.setStyleSheet(get_adaptive_stylesheet("font-size: 11pt; color: #666;"))
        self.main_layout.addWidget(test_desc, alignment=Qt.AlignCenter)

        self.main_layout.addStretch(1) # Add stretch to push content to the center vertically

        self.apply_styles()

    def apply_styles(self):
        # Style for "Start New Training" button - light green
        new_training_button_style = get_adaptive_stylesheet("""
            QPushButton {
                font-size: 16pt;
                padding: 18px 35px;
                border-radius: 8px;
                background-color: #28a745;
                color: white;
                font-weight: bold;
                min-width: 350px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:disabled {
                background-color: #d3d3d3;
                color: #a9a9a9;
            }
        """)
        
        # Style for "Test Trained Model" button - dark orange
        test_model_button_style = get_adaptive_stylesheet("""
            QPushButton {
                font-size: 16pt;
                padding: 18px 35px;
                border-radius: 8px;
                background-color: #cc5500;
                color: white;
                font-weight: bold;
                min-width: 350px;
            }
            QPushButton:hover {
                background-color: #b84700;
            }
            QPushButton:disabled {
                background-color: #d3d3d3;
                color: #a9a9a9;
            }
        """)
        
        self.new_training_button.setStyleSheet(new_training_button_style)
        self.test_model_button.setStyleSheet(test_model_button_style)
        self.central_widget.setStyleSheet("background-color: #f8f9fa;")

    def start_new_training(self):
        self.data_import_gui = DataImportGUI()
        self.data_import_gui.show()
        self.close()

    def test_trained_model(self):
        self.test_selection_gui = TestSelectionGUI() 
        self.test_selection_gui.show()
        self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    welcome_screen = WelcomeGUI()
    welcome_screen.show()
    sys.exit(app.exec_())