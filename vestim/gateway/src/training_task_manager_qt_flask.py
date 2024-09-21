from flask import Flask, jsonify, request, Blueprint

import time, os, sys
import csv
import sqlite3
import torch
from vestim.gateway.src.job_manager_qt import JobManager
from vestim.gateway.src.training_setup_manager_qt_flask import VEstimTrainingSetupManager
from vestim.services.model_training.src.data_loader_service_padfil import DataLoaderService
from vestim.services.model_training.src.training_task_service_test import TrainingTaskService
import logging


# Training Task Manager Blueprint
training_task_blueprint = Blueprint('training_task_manager', __name__)

class TrainingTaskManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.job_manager = JobManager()
        self.data_loader_service = DataLoaderService()
        self.training_service = TrainingTaskService()
        self.training_setup_manager = VEstimTrainingSetupManager()
        self.current_task = None
        self.stop_requested = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_thread = None

    def log_to_csv(self, task, epoch, train_loss, val_loss, elapsed_time, current_lr, best_val_loss, delta_t_epoch):
        csv_log_file = task['csv_log_file']
        with open(csv_log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Epoch', 'Train Loss', 'Val Loss', 'Elapsed Time', 'Learning Rate', 'Best Val Loss', 'Train Time Per Epoch'])
            writer.writerow({
                'Epoch': epoch,
                'Train Loss': train_loss,
                'Val Loss': val_loss,
                'Elapsed Time': elapsed_time,
                'Learning Rate': current_lr,
                'Best Val Loss': best_val_loss,
                'Train Time Per Epoch': delta_t_epoch
            })

    def log_to_sqlite(self, task, epoch, train_loss, val_loss, best_val_loss, elapsed_time, avg_batch_time, early_stopping, model_memory_usage):
        sqlite_db_file = task['db_log_file']
        conn = sqlite3.connect(sqlite_db_file)
        cursor = conn.cursor()

        cursor.execute('''INSERT INTO task_logs (task_id, epoch, train_loss, val_loss, elapsed_time, avg_batch_time, learning_rate, 
                        best_val_loss, num_learnable_params, batch_size, lookback, max_epochs, early_stopping, model_memory_usage, device)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (task['task_id'], epoch, train_loss, val_loss, elapsed_time, avg_batch_time, task['hyperparams']['INITIAL_LR'], best_val_loss,
                        task['hyperparams']['NUM_LEARNABLE_PARAMS'], task['hyperparams']['BATCH_SIZE'], task['hyperparams']['LOOKBACK'],task['hyperparams']['MAX_EPOCHS'],
                        early_stopping, model_memory_usage, self.device.type))

        conn.commit()
        conn.close()

    def setup_job_logging(self, task):
        job_id = self.job_manager.get_job_id()
        model_dir = task.get('model_dir')
        csv_log_file = task.get('csv_log_file')
        db_log_file = task.get('db_log_file')

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        self.create_sql_tables(db_log_file)

    def create_sql_tables(self, db_log_file):
        try:
            if not os.path.isfile(db_log_file):
                self.logger.info(f"Creating new database file at: {db_log_file}")

            conn = sqlite3.connect(db_log_file)
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS task_logs (
                    task_id TEXT, epoch INTEGER, train_loss REAL, val_loss REAL, elapsed_time REAL,
                    avg_batch_time REAL, learning_rate REAL, best_val_loss REAL, num_learnable_params INTEGER,
                    batch_size INTEGER, lookback INTEGER, max_epochs INTEGER, early_stopping INTEGER, 
                    model_memory_usage REAL, device TEXT, PRIMARY KEY(task_id, epoch)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS batch_logs (
                    task_id TEXT, epoch INTEGER, batch_idx INTEGER, batch_time REAL, phase TEXT,
                    learning_rate REAL, num_learnable_params INTEGER, batch_size INTEGER, lookback INTEGER,
                    device TEXT, FOREIGN KEY(task_id, epoch) REFERENCES task_logs(task_id, epoch)
                )
            ''')

            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            self.logger.error(f"SQLite error: {e}")
            raise e

    def run_training(self, task):
        try:
            # Training logic remains the same as before
            self.logger.info("Starting training loop")
            # Additional training logic goes here
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")

# Flask Endpoints for Training Task Manager

@training_task_blueprint.route('/start_task', methods=['POST'])
def start_task():
    task = request.json.get('task')
    update_progress_callback = request.json.get('update_progress_callback')  # Placeholder for the callback

    if not task:
        return jsonify({'error': 'No task data provided'}), 400

    try:
        task_manager = TrainingTaskManager()
        task_manager.process_task(task, update_progress_callback)
        return jsonify({'message': 'Task started successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@training_task_blueprint.route('/stop_task', methods=['POST'])
def stop_task():
    try:
        task_manager = TrainingTaskManager()
        task_manager.stop_task()
        return jsonify({'message': 'Task stopped successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500



