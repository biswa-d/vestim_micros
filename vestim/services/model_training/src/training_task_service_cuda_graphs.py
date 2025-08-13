# Enhanced Training Service with CUDA Graphs Optimization
# This provides significant speedups for small/medium FNNs on Windows with NVIDIA GPUs

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import csv
import sqlite3
import os
from typing import Optional, Tuple, List, Union


class CUDAGraphsTrainingService:
    """
    Enhanced training service with CUDA Graphs optimization.
    
    Benefits for your RTX 5070 setup:
    - 1.2x-3x speedup for small/medium FNNs 
    - Higher GPU utilization 
    - Reduced CPU-GPU launch overhead (especially on Windows)
    """
    
    def __init__(self):
        self.criterion = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # CUDA Graphs related attributes
        self.cuda_graph = None
        self.static_input = None
        self.static_target = None
        self.static_output = None
        self.static_loss = None
        self.graph_captured = False
        self.current_batch_size = None
        
        # Performance tracking
        self.cuda_graph_enabled = False
        self.warmup_steps = 3  # Number of warmup steps before capturing graph
        
        print(f"CUDAGraphsTrainingService initialized on device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"CUDA Graphs will be used for FNN models to accelerate training")

    def _should_use_cuda_graphs(self, model_type: str, device: torch.device, batch_size: int) -> bool:
        """
        Determine if CUDA Graphs should be used based on model type and conditions.
        
        CUDA Graphs work best for:
        - FNN models (many small kernels)
        - CUDA devices
        - Fixed batch sizes
        - Small to medium models (launch-limited rather than compute-limited)
        """
        if device.type != 'cuda':
            return False
            
        if model_type != 'FNN':  # For now, only optimize FNNs
            return False
            
        if batch_size is None or batch_size <= 0:
            return False
            
        # Additional checks could include model parameter count, etc.
        return True

    def _warmup_model(self, model, model_type: str, sample_input, sample_target, 
                     optimizer, use_mixed_precision: bool, scaler=None):
        """
        Warm up the model to allocate all necessary memory buffers before graph capture.
        This ensures no memory allocations happen during graph execution.
        """
        model.train()
        
        for _ in range(self.warmup_steps):
            optimizer.zero_grad(set_to_none=True)
            
            if use_mixed_precision and scaler is not None:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    if model_type == "FNN":
                        output = model(sample_input)
                    else:
                        raise ValueError(f"CUDA Graphs not yet supported for {model_type}")
                    
                    loss = self.criterion(output, sample_target)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                if model_type == "FNN":
                    output = model(sample_input)
                else:
                    raise ValueError(f"CUDA Graphs not yet supported for {model_type}")
                
                loss = self.criterion(output, sample_target)
                loss.backward()
                optimizer.step()
        
        torch.cuda.synchronize()
        print(f"Model warmed up with {self.warmup_steps} steps")

    def _capture_cuda_graph(self, model, model_type: str, batch_size: int, input_size: int,
                           optimizer, use_mixed_precision: bool, scaler=None):
        """
        Capture the training step into a CUDA Graph for repeated execution.
        """
        print(f"Capturing CUDA Graph for {model_type} model...")
        
        # Create static tensors with the exact shapes we'll use
        if model_type == "FNN":
            self.static_input = torch.randn(batch_size, input_size, device=self.device)
            self.static_target = torch.randn(batch_size, 1, device=self.device)  # Assuming single output
        else:
            raise ValueError(f"CUDA Graphs not yet supported for {model_type}")
        
        # Warmup first to allocate all buffers
        self._warmup_model(model, model_type, self.static_input, self.static_target, 
                          optimizer, use_mixed_precision, scaler)
        
        # Create and capture the graph
        self.cuda_graph = torch.cuda.CUDAGraph()
        
        # Zero gradients before capture (required)
        optimizer.zero_grad(set_to_none=True)
        
        # Capture the forward + backward + optimizer step
        with torch.cuda.graph(self.cuda_graph):
            if use_mixed_precision and scaler is not None:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    self.static_output = model(self.static_input)
                    self.static_loss = self.criterion(self.static_output, self.static_target)
                
                scaler.scale(self.static_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                self.static_output = model(self.static_input)
                self.static_loss = self.criterion(self.static_output, self.static_target)
                self.static_loss.backward()
                optimizer.step()
        
        self.graph_captured = True
        self.current_batch_size = batch_size
        self.cuda_graph_enabled = True
        
        print(f"✅ CUDA Graph captured successfully!")
        print(f"   Expected speedup: 1.2x-3x for small kernels on Windows")
        print(f"   Batch size: {batch_size}, Input size: {input_size}")

    def _execute_cuda_graph_step(self, X_batch, y_batch):
        """
        Execute one training step using the captured CUDA Graph.
        This is much faster than individual kernel launches.
        """
        # Copy real data into static tensors
        self.static_input.copy_(X_batch, non_blocking=True)
        self.static_target.copy_(y_batch, non_blocking=True)
        
        # Replay the entire training step with one GPU command
        self.cuda_graph.replay()
        
        # Return the results
        return self.static_output.clone(), self.static_loss.item()

    def train_epoch(self, model, model_type, train_loader, optimizer, h_s_initial, h_c_initial, 
                   epoch, device, stop_requested, task, verbose=True):
        """
        Enhanced train_epoch with CUDA Graphs optimization for FNN models.
        
        For FNN models on CUDA: Uses CUDA Graphs for ~1.2x-3x speedup
        For other models: Falls back to standard training
        """
        model.train()
        total_train_loss = []
        all_train_y_pred_normalized = []
        all_train_y_true_normalized = []
        batch_times = []
        log_freq = task.get('log_frequency', 100)
        
        # Mixed precision setup
        use_mixed_precision = task['hyperparams'].get('USE_MIXED_PRECISION', False) and device.type == 'cuda'
        scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
        
        # Enable high-performance matrix multiplication on RTX 50 series
        if device.type == 'cuda' and hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')  # Enables TensorFloat-32 on Ampere+
        
        if use_mixed_precision:
            print(f"🚀 Using mixed precision (AMP) + TF32 for epoch {epoch}")
        
        # Get a sample batch to determine if we can use CUDA Graphs
        sample_batch = next(iter(train_loader))
        sample_X, sample_y = sample_batch[0], sample_batch[1]
        batch_size = sample_X.size(0)
        input_size = sample_X.size(1) if len(sample_X.shape) == 2 else sample_X.numel() // batch_size
        
        # Check if we should use CUDA Graphs
        should_use_graphs = self._should_use_cuda_graphs(model_type, device, batch_size)
        
        # Capture CUDA Graph if conditions are met and not already captured
        if should_use_graphs and not self.graph_captured and model_type == "FNN":
            try:
                sample_X_gpu = sample_X.to(device, non_blocking=True)
                sample_y_gpu = sample_y.to(device, non_blocking=True)
                
                # Handle target shape compatibility
                if sample_y_gpu.ndim == 1 and len(sample_y_gpu) == batch_size:
                    sample_y_gpu = sample_y_gpu.unsqueeze(1)
                
                self._capture_cuda_graph(model, model_type, batch_size, input_size,
                                       optimizer, use_mixed_precision, scaler)
                print(f"✅ CUDA Graphs enabled for FNN training (RTX 5070 optimized)")
                
            except Exception as e:
                print(f"⚠️  CUDA Graph capture failed, falling back to standard training: {e}")
                should_use_graphs = False
                self.cuda_graph_enabled = False
        
        # Training loop - with or without CUDA Graphs
        cuda_events_enabled = device.type == 'cuda'
        if cuda_events_enabled:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            if stop_requested:
                print("Stop requested during training")
                break
            
            # Time the batch processing
            if cuda_events_enabled:
                start_event.record()
            else:
                start_batch_time = time.time()
            
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            
            # Handle target shape compatibility
            if y_batch.ndim == 1 and len(y_batch) == batch_size:
                y_batch = y_batch.unsqueeze(1)
            
            # Execute training step
            if self.cuda_graph_enabled and model_type == "FNN" and X_batch.size(0) == self.current_batch_size:
                # Use CUDA Graph for maximum speed
                y_pred, loss_value = self._execute_cuda_graph_step(X_batch, y_batch)
                loss = torch.tensor(loss_value, device=device)  # Create tensor for compatibility
                
            else:
                # Standard training path (RNNs, variable batch sizes, etc.)
                optimizer.zero_grad()
                
                if use_mixed_precision:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        if model_type == "LSTM":
                            if h_s_initial is None:
                                h_s = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units, device=device)
                                h_c = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units, device=device)
                            else:
                                h_s = h_s_initial.detach().clone()
                                h_c = h_c_initial.detach().clone()
                            y_pred, (h_s, h_c) = model(X_batch, h_s, h_c)
                            
                        elif model_type == "GRU":
                            if h_s_initial is None:
                                h_s = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units, device=device)
                            else:
                                h_s = h_s_initial.detach().clone()
                            y_pred, h_s = model(X_batch, h_s)
                            
                        elif model_type == "FNN":
                            y_pred = model(X_batch)
                        else:
                            raise ValueError(f"Unsupported model_type: {model_type}")
                        
                        # Handle output shape compatibility
                        if y_pred.ndim > y_batch.ndim and y_pred.shape[-1] == 1 and y_batch.ndim >= 1:
                            y_pred = y_pred.squeeze(-1)
                        
                        loss = self.criterion(y_pred, y_batch)
                    
                    scaler.scale(loss).backward()
                    
                    if model_type in ["LSTM", "GRU"]:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    
                else:
                    # Standard precision
                    if model_type == "LSTM":
                        if h_s_initial is None:
                            h_s = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units, device=device)
                            h_c = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units, device=device)
                        else:
                            h_s = h_s_initial.detach().clone()
                            h_c = h_c_initial.detach().clone()
                        y_pred, (h_s, h_c) = model(X_batch, h_s, h_c)
                        
                    elif model_type == "GRU":
                        if h_s_initial is None:
                            h_s = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units, device=device)
                        else:
                            h_s = h_s_initial.detach().clone()
                        y_pred, h_s = model(X_batch, h_s)
                        
                    elif model_type == "FNN":
                        y_pred = model(X_batch)
                    else:
                        raise ValueError(f"Unsupported model_type: {model_type}")
                    
                    # Handle output shape compatibility
                    if y_pred.ndim > y_batch.ndim and y_pred.shape[-1] == 1 and y_batch.ndim >= 1:
                        y_pred = y_pred.squeeze(-1)
                    
                    loss = self.criterion(y_pred, y_batch)
                    
                    if model_type in ["LSTM", "GRU"]:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    loss.backward()
                    optimizer.step()
            
            # Record timing and metrics
            if cuda_events_enabled:
                end_event.record()
                torch.cuda.synchronize()
                batch_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
            else:
                batch_time = time.time() - start_batch_time
            
            batch_times.append(batch_time)
            total_train_loss.append(loss.item())
            
            # Store predictions and true values
            all_train_y_pred_normalized.append(y_pred.detach().cpu())
            all_train_y_true_normalized.append(y_batch.detach().cpu())
            
            # Logging
            if verbose and batch_idx % log_freq == 0 and batch_times:
                graph_status = "🚀 CUDA Graph" if (self.cuda_graph_enabled and X_batch.size(0) == self.current_batch_size) else "Standard"
                log_callback = task.get('log_callback')
                if log_callback:
                    log_callback(f"  Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f} [{graph_status}]")
                else:
                    print(f"Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Time: {batch_time*1000:.2f}ms [{graph_status}]")
            
            # Cleanup (less aggressive for CUDA Graphs)
            if not self.cuda_graph_enabled:
                del X_batch, y_batch, y_pred
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        # Calculate statistics
        avg_epoch_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
        avg_loss = sum(total_train_loss) / len(total_train_loss) if total_train_loss else float('nan')
        
        # Performance summary
        if verbose and self.cuda_graph_enabled:
            total_batches = len(batch_times)
            total_time = sum(batch_times)
            avg_time_ms = avg_epoch_batch_time * 1000
            print(f"🚀 CUDA Graph Performance: {total_batches} batches in {total_time:.3f}s ({avg_time_ms:.2f}ms/batch)")
        
        # Concatenate results
        if all_train_y_pred_normalized:
            all_train_y_pred_normalized = torch.cat(all_train_y_pred_normalized, dim=0)
        if all_train_y_true_normalized:
            all_train_y_true_normalized = torch.cat(all_train_y_true_normalized, dim=0)
        
        return avg_epoch_batch_time, avg_loss, all_train_y_pred_normalized, all_train_y_true_normalized

    def validate_epoch(self, model, model_type, val_loader, h_s_initial, h_c_initial, 
                      epoch, device, stop_requested, task, verbose=True):
        """
        Standard validation epoch (CUDA Graphs not needed for validation due to no backward pass).
        """
        model.eval()
        total_val_loss = []
        all_val_y_pred_normalized = []
        all_val_y_true_normalized = []
        log_freq = task.get('log_frequency', 100)
        
        use_mixed_precision = task['hyperparams'].get('USE_MIXED_PRECISION', False) and device.type == 'cuda'
        
        with torch.no_grad():
            for batch_idx, (X_batch, y_batch) in enumerate(val_loader):
                if stop_requested:
                    break
                
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                
                if use_mixed_precision:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        if model_type == "LSTM":
                            if h_s_initial is None:
                                h_s = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units, device=device)
                                h_c = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units, device=device)
                            else:
                                h_s = h_s_initial.detach().clone()
                                h_c = h_c_initial.detach().clone()
                            y_pred, _ = model(X_batch, h_s, h_c)
                            
                        elif model_type == "GRU":
                            if h_s_initial is None:
                                h_s = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units, device=device)
                            else:
                                h_s = h_s_initial.detach().clone()
                            y_pred, _ = model(X_batch, h_s)
                            
                        elif model_type == "FNN":
                            y_pred = model(X_batch)
                        else:
                            raise ValueError(f"Unsupported model_type: {model_type}")
                        
                        # Handle shape compatibility
                        if y_pred.ndim > y_batch.ndim and y_pred.shape[-1] == 1:
                            y_pred = y_pred.squeeze(-1)
                        if y_batch.ndim == 1 and y_pred.ndim == 2 and y_pred.shape[1] == 1:
                            y_batch = y_batch.unsqueeze(1)
                        
                        loss = self.criterion(y_pred, y_batch)
                else:
                    # Standard precision validation
                    if model_type == "LSTM":
                        if h_s_initial is None:
                            h_s = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units, device=device)
                            h_c = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units, device=device)
                        else:
                            h_s = h_s_initial.detach().clone()
                            h_c = h_c_initial.detach().clone()
                        y_pred, _ = model(X_batch, h_s, h_c)
                        
                    elif model_type == "GRU":
                        if h_s_initial is None:
                            h_s = torch.zeros(model.num_layers, X_batch.size(0), model.hidden_units, device=device)
                        else:
                            h_s = h_s_initial.detach().clone()
                        y_pred, _ = model(X_batch, h_s)
                        
                    elif model_type == "FNN":
                        y_pred = model(X_batch)
                    else:
                        raise ValueError(f"Unsupported model_type: {model_type}")
                    
                    # Handle shape compatibility
                    if y_pred.ndim > y_batch.ndim and y_pred.shape[-1] == 1:
                        y_pred = y_pred.squeeze(-1)
                    if y_batch.ndim == 1 and y_pred.ndim == 2 and y_pred.shape[1] == 1:
                        y_batch = y_batch.unsqueeze(1)
                    
                    loss = self.criterion(y_pred, y_batch)
                
                total_val_loss.append(loss.item())
                all_val_y_pred_normalized.append(y_pred.cpu())
                all_val_y_true_normalized.append(y_batch.cpu())
                
                if verbose and batch_idx % log_freq == 0:
                    print(f"Validation - Epoch: {epoch}, Batch: {batch_idx}/{len(val_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = sum(total_val_loss) / len(total_val_loss) if total_val_loss else float('nan')
        
        if all_val_y_pred_normalized:
            all_val_y_pred_normalized = torch.cat(all_val_y_pred_normalized, dim=0)
        if all_val_y_true_normalized:
            all_val_y_true_normalized = torch.cat(all_val_y_true_normalized, dim=0)
        
        return avg_loss, all_val_y_pred_normalized, all_val_y_true_normalized

    def reset_cuda_graph(self):
        """Reset CUDA Graph state (useful when changing batch sizes or model architecture)."""
        self.cuda_graph = None
        self.static_input = None
        self.static_target = None
        self.static_output = None
        self.static_loss = None
        self.graph_captured = False
        self.current_batch_size = None
        self.cuda_graph_enabled = False
        print("CUDA Graph state reset")

    def get_performance_stats(self) -> dict:
        """Get performance statistics."""
        return {
            'cuda_graphs_enabled': self.cuda_graph_enabled,
            'device': str(self.device),
            'graph_captured': self.graph_captured,
            'current_batch_size': self.current_batch_size,
        }

    # Legacy methods for compatibility with existing code
    def log_to_csv(self, task, epoch, batch_idx, batch_time, phase):
        """Log batch timing data to a CSV file."""
        csv_log_file = task['csv_log_file']
        fieldnames = ['Epoch', 'Batch', 'Batch Time', 'Phase', 'CUDA_Graph_Used']
        file_exists = os.path.isfile(csv_log_file)

        with open(csv_log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'Epoch': epoch,
                'Batch': batch_idx,
                'Batch Time': batch_time,
                'Phase': phase,
                'CUDA_Graph_Used': self.cuda_graph_enabled
            })

    def log_to_sqlite(self, task, epoch, batch_idx, batch_time, phase, device):
        """Log batch timing data to a SQLite database."""
        sqlite_db_file = task['db_log_file']
        conn = sqlite3.connect(sqlite_db_file)
        cursor = conn.cursor()

        cursor.execute('''INSERT INTO batch_logs 
                         (task_id, epoch, batch_idx, batch_time, phase, learning_rate, 
                          num_learnable_params, batch_size, lookback, device)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                      (task['task_id'], epoch, batch_idx, batch_time, phase, 
                       task['hyperparams']['INITIAL_LR'], task['hyperparams']['NUM_LEARNABLE_PARAMS'],
                       task['hyperparams']['BATCH_SIZE'], task['hyperparams']['LOOKBACK'], str(device)))

        conn.commit()
        conn.close()
