import torch
import torch.nn as nn
import time
import logging
from typing import Tuple, Optional


class CUDAGraphsTrainingService:
    """Enhanced training service with CUDA Graphs optimization for faster training."""
    
    def __init__(self, device=None):
        self.logger = logging.getLogger(__name__)
        self.criterion = nn.MSELoss()
        self.graphs_enabled = False
        self.train_graph = None
        self.val_graph = None
        self.static_input = None
        self.static_target = None
        self.static_output = None
        self.static_loss = None
        self.static_val_input = None
        self.static_val_target = None
        self.scaler = None
        
        # Store the device - either passed in or auto-detected
        if device is not None:
            self.device = device
            self.logger.info(f"CUDAGraphsTrainingService: Using specified device: {device}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"CUDAGraphsTrainingService: Auto-detected device: {self.device}")
        
    def enable_cuda_graphs(self, device, use_mixed_precision=True):
        """Enable CUDA Graphs if supported."""
        if device.type != 'cuda':
            self.logger.info("CUDA Graphs requires CUDA device. Falling back to standard training.")
            return False
            
        if not torch.cuda.is_available():
            self.logger.info("CUDA not available. Falling back to standard training.")
            return False
            
        self.graphs_enabled = True
        self.scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
        self.logger.info(f"CUDA Graphs enabled on {device}")
        return True
    
    def reset_cuda_graphs(self):
        """Reset CUDA graphs state - call this after model reloading."""
        if self.graphs_enabled:
            self.train_graph = None
            self.val_graph = None
            self.static_input = None
            self.static_target = None
            self.static_output = None
            self.static_loss = None
            self.static_val_input = None
            self.static_val_target = None
            self.logger.info("CUDA graphs state reset - graphs will be re-captured on next training epoch")
    
    def _warmup_model(self, model, sample_input, sample_target, optimizer, device):
        """Warmup to allocate all memory before capturing graph."""
        self.logger.info("Warming up model for CUDA Graphs...")
        
        # Warmup forward and backward pass
        optimizer.zero_grad(set_to_none=True)
        
        if self.scaler:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                output = model(sample_input)
                if output.shape != sample_target.shape:
                    if output.ndim > sample_target.ndim and output.shape[-1] == 1:
                        output = output.squeeze(-1)
                    elif sample_target.ndim == 1 and output.ndim == 2 and output.shape[1] == 1:
                        sample_target = sample_target.unsqueeze(1)
                loss = self.criterion(output, sample_target)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            output = model(sample_input)
            if output.shape != sample_target.shape:
                if output.ndim > sample_target.ndim and output.shape[-1] == 1:
                    output = output.squeeze(-1)
                elif sample_target.ndim == 1 and output.ndim == 2 and output.shape[1] == 1:
                    sample_target = sample_target.unsqueeze(1)
            loss = self.criterion(output, sample_target)
            loss.backward()
            optimizer.step()
        
        torch.cuda.synchronize()
        self.logger.info("Model warmup completed")
    
    def _capture_training_graph(self, model, optimizer, device, input_shape, target_shape):
        """Capture the training step as a CUDA graph."""
        self.logger.info("Capturing training CUDA graph...")
        
        # Create static tensors with the same shapes
        self.static_input = torch.empty(input_shape, device=device, dtype=torch.float32)
        self.static_target = torch.empty(target_shape, device=device, dtype=torch.float32)
        
        # Prepare for graph capture
        self.train_graph = torch.cuda.CUDAGraph()
        # Capture the graph
        with torch.cuda.graph(self.train_graph):
            optimizer.zero_grad(set_to_none=True)
            if self.scaler:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    self.static_output = model(self.static_input)
                    # Handle shape mismatch
                    if self.static_output.shape != self.static_target.shape:
                        if (self.static_output.ndim > self.static_target.ndim and 
                            self.static_output.shape[-1] == 1):
                            self.static_output = self.static_output.squeeze(-1)
                    self.static_loss = self.criterion(self.static_output, self.static_target)
                
                # The optimizer step must be captured inside the graph
                # The scaler manages the gradient scaling and optimizer step together
                self.scaler.scale(self.static_loss).backward()
                # We capture the scaler.step(optimizer) which includes the weight update
                self.scaler.step(optimizer)
                # The update to the scaler's scale factor should be done outside the graph
            else:
                self.static_output = model(self.static_input)
                if self.static_output.shape != self.static_target.shape:
                    if (self.static_output.ndim > self.static_target.ndim and
                        self.static_output.shape[-1] == 1):
                        self.static_output = self.static_output.squeeze(-1)
                self.static_loss = self.criterion(self.static_output, self.static_target)
                self.static_loss.backward()
                optimizer.step() # This is correct for no scaler
        
        self.logger.info("Training CUDA graph captured successfully")
    
    def train_epoch_with_graphs(self, model, train_loader, optimizer, epoch, device, 
                               stop_requested, task, verbose=True):
        """Train one epoch using CUDA graphs for maximum performance."""
        model.train()
        total_train_loss = []
        all_train_y_pred = []
        all_train_y_true = []
        batch_times = []
        
        # Get first batch for initialization
        first_batch = next(iter(train_loader))
        sample_input, sample_target = first_batch[0].to(device), first_batch[1].to(device)
        
        # Warmup and capture graph on first epoch or after reset
        if self.train_graph is None:
            self._warmup_model(model, sample_input, sample_target, optimizer, device)
            self._capture_training_graph(model, optimizer, device, 
                                       sample_input.shape, sample_target.shape)
        
        # Training loop with CUDA graphs
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            if stop_requested:
                self.logger.info("Training stopped by user request")
                break
            
            start_time = time.time()
            X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            
            # Check if batch shape matches captured graph (or if graphs need to be re-captured)
            if (self.static_input is None or self.static_target is None or
                X_batch.shape != self.static_input.shape or 
                y_batch.shape != self.static_target.shape):
                if self.static_input is None or self.static_target is None:
                    # Only log once per epoch when graphs need re-capturing
                    if batch_idx == 0:
                        self.logger.info("CUDA graphs not captured yet - will capture before training loop")
                else:
                    self.logger.warning(f"Batch shape mismatch at batch {batch_idx}. "
                                      f"Expected input: {self.static_input.shape}, got: {X_batch.shape}. "
                                      f"Expected target: {self.static_target.shape}, got: {y_batch.shape}. "
                                      f"Falling back to standard training for this batch.")
                # Fall back to standard training for this batch
                optimizer.zero_grad()
                if self.scaler:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        output = model(X_batch)
                        if output.shape != y_batch.shape and output.ndim > y_batch.ndim:
                            output = output.squeeze(-1)
                        loss = self.criterion(output, y_batch)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    output = model(X_batch)
                    if output.shape != y_batch.shape and output.ndim > y_batch.ndim:
                        output = output.squeeze(-1)
                    loss = self.criterion(output, y_batch)
                    loss.backward()
                    optimizer.step()
            else:
                # Use CUDA graphs - copy data and replay
                self.static_input.copy_(X_batch)
                self.static_target.copy_(y_batch)
                self.train_graph.replay()
                
                # After replaying the graph, if using a scaler, we need to update it
                if self.scaler:
                    self.scaler.update()
                
                # Get the loss and output from the static tensors
                loss = self.static_loss
                output = self.static_output
            
            # Collect metrics
            total_train_loss.append(loss.item())
            all_train_y_pred.append(output.detach().cpu())
            all_train_y_true.append(y_batch.detach().cpu())
            
            batch_times.append(time.time() - start_time)
            
            if verbose and batch_idx % 100 == 0:
                self.logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, "
                               f"Loss: {loss.item():.6f}, Time: {batch_times[-1]*1000:.1f}ms")
        
        # Calculate metrics
        avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
        avg_loss = sum(total_train_loss) / len(total_train_loss) if total_train_loss else float('nan')
        
        # Concatenate predictions
        all_train_y_pred = torch.cat(all_train_y_pred, dim=0) if all_train_y_pred else None
        all_train_y_true = torch.cat(all_train_y_true, dim=0) if all_train_y_true else None
        
        self.logger.info(f"Epoch {epoch} completed. Avg batch time: {avg_batch_time*1000:.1f}ms, "
                        f"Avg loss: {avg_loss:.6f}")
        
        return avg_batch_time, avg_loss, all_train_y_pred, all_train_y_true
    
    def validate_epoch_with_graphs(self, model, val_loader, epoch, device, stop_requested, task, verbose=True):
        """Validate with potential CUDA graph optimization."""
        model.eval()
        total_val_loss = []
        all_val_y_pred = []
        all_val_y_true = []
        batch_times = []
        
        with torch.no_grad():
            for batch_idx, (X_batch, y_batch) in enumerate(val_loader):
                if stop_requested:
                    break
                
                start_time = time.time()
                X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
                
                if self.scaler:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        output = model(X_batch)
                        if output.shape != y_batch.shape and output.ndim > y_batch.ndim:
                            output = output.squeeze(-1)
                        loss = self.criterion(output, y_batch)
                else:
                    output = model(X_batch)
                    if output.shape != y_batch.shape and output.ndim > y_batch.ndim:
                        output = output.squeeze(-1)
                    loss = self.criterion(output, y_batch)
                
                total_val_loss.append(loss.item())
                all_val_y_pred.append(output.cpu())
                all_val_y_true.append(y_batch.cpu())
                batch_times.append(time.time() - start_time)

                if verbose and batch_idx % 100 == 0:
                    self.logger.info(f"Epoch {epoch}, Validation Batch {batch_idx}/{len(val_loader)}, "
                                   f"Loss: {loss.item():.6f}, Time: {batch_times[-1]*1000:.1f}ms")
        
        avg_loss = sum(total_val_loss) / len(total_val_loss) if total_val_loss else float('nan')
        all_val_y_pred = torch.cat(all_val_y_pred, dim=0) if all_val_y_pred else None
        all_val_y_true = torch.cat(all_val_y_true, dim=0) if all_val_y_true else None
        
        return avg_loss, all_val_y_pred, all_val_y_true
    
    def validate_epoch(self, model, model_type, val_loader, h_s_val, h_c_val, epoch, device, stop_requested, task, verbose=True):
        """Fallback validate_epoch method for compatibility with standard training code."""
        # Just delegate to validate_epoch_with_graphs since they're essentially the same for validation
        return self.validate_epoch_with_graphs(model, val_loader, epoch, device, stop_requested, task, verbose=verbose)
