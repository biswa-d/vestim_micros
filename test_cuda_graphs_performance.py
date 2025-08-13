"""
Quick CUDA Graphs Performance Test for RTX 5070
Run this script to see the immediate difference CUDA Graphs makes!
"""
import torch
import torch.nn as nn
import time
import numpy as np


def create_test_fnn_model(input_size=50, hidden_sizes=[256, 128, 64], output_size=1):
    """Create a small FNN model for testing."""
    layers = []
    prev_size = input_size
    
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(prev_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))
        prev_size = hidden_size
    
    layers.append(nn.Linear(prev_size, output_size))
    return nn.Sequential(*layers)


def time_standard_training(model, data_loader, optimizer, device, num_steps=100):
    """Time standard training approach."""
    model.train()
    criterion = nn.MSELoss()
    
    times = []
    step_count = 0
    
    print("🐌 Testing standard training...")
    
    for batch_idx, (x, y) in enumerate(data_loader):
        if step_count >= num_steps:
            break
            
        x, y = x.to(device), y.to(device)
        
        start_time = time.perf_counter()
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)  # Convert to ms
        step_count += 1
    
    avg_time = np.mean(times[10:])  # Skip first 10 for warmup
    print(f"   Average step time: {avg_time:.2f}ms")
    return avg_time


def time_cuda_graphs_training(model, data_loader, optimizer, device, num_steps=100):
    """Time CUDA Graphs training approach."""
    model.train()
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()
    
    # Get sample batch for graph capture
    sample_x, sample_y = next(iter(data_loader))
    sample_x, sample_y = sample_x.to(device), sample_y.to(device)
    
    print("🚀 Testing CUDA Graphs training...")
    print("   Warming up model...")
    
    # Warmup
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        output = model(sample_x)
        loss = criterion(output, sample_y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    
    print("   Capturing CUDA graph...")
    
    # Create static tensors
    static_x = torch.empty_like(sample_x)
    static_y = torch.empty_like(sample_y)
    
    # Capture graph
    graph = torch.cuda.CUDAGraph()
    optimizer.zero_grad(set_to_none=True)
    
    with torch.cuda.graph(graph):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            static_output = model(static_x)
            static_loss = criterion(static_output, static_y)
        scaler.scale(static_loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    print("   Running timed training loop...")
    
    # Timed training loop
    times = []
    step_count = 0
    
    for batch_idx, (x, y) in enumerate(data_loader):
        if step_count >= num_steps:
            break
            
        x, y = x.to(device), y.to(device)
        
        start_time = time.perf_counter()
        
        # Copy data and replay graph
        static_x.copy_(x)
        static_y.copy_(y)
        graph.replay()
        
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)  # Convert to ms
        step_count += 1
    
    avg_time = np.mean(times[10:])  # Skip first 10 for warmup
    print(f"   Average step time: {avg_time:.2f}ms")
    return avg_time


def main():
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This test requires a CUDA GPU.")
        return
    
    device = torch.device('cuda:0')
    print(f"🎯 Running performance test on: {torch.cuda.get_device_name(device)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    
    # Create test model and data
    model = create_test_fnn_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
    
    # Create synthetic dataset
    batch_size = 1024
    num_batches = 150
    input_size = 50
    output_size = 1
    
    print(f"\n📊 Test Configuration:")
    print(f"   Model: FNN with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"   Batch size: {batch_size}")
    print(f"   Steps to time: 100")
    print(f"   Mixed precision: bfloat16")
    
    # Generate data
    data = []
    for _ in range(num_batches):
        x = torch.randn(batch_size, input_size)
        y = torch.randn(batch_size, output_size)
        data.append((x, y))
    
    print("\n" + "="*60)
    
    # Test standard training
    model_copy1 = create_test_fnn_model().to(device)
    model_copy1.load_state_dict(model.state_dict())
    optimizer1 = torch.optim.AdamW(model_copy1.parameters(), lr=1e-3, fused=True)
    
    standard_time = time_standard_training(model_copy1, data, optimizer1, device)
    
    print("\n" + "-"*40)
    
    # Test CUDA Graphs training
    model_copy2 = create_test_fnn_model().to(device)
    model_copy2.load_state_dict(model.state_dict())
    optimizer2 = torch.optim.AdamW(model_copy2.parameters(), lr=1e-3, fused=True)
    
    graphs_time = time_cuda_graphs_training(model_copy2, data, optimizer2, device)
    
    print("\n" + "="*60)
    
    # Calculate speedup
    speedup = standard_time / graphs_time
    time_saved_per_epoch = (standard_time - graphs_time) * 100  # Assuming 100 steps per epoch
    
    print(f"📈 RESULTS:")
    print(f"   Standard training: {standard_time:.2f}ms per step")
    print(f"   CUDA Graphs:       {graphs_time:.2f}ms per step")
    print(f"   🚀 Speedup:        {speedup:.2f}x faster")
    print(f"   ⏱️  Time saved:     {time_saved_per_epoch/1000:.1f}s per 100-step epoch")
    
    if speedup > 1.2:
        print(f"\n✅ Great! CUDA Graphs provides {speedup:.1f}x speedup on your RTX 5070!")
        print("   This will significantly reduce your training time.")
    else:
        print(f"\n⚠️  Speedup is modest ({speedup:.1f}x). Try larger batch sizes or different models.")
    
    print(f"\nTo enable in your training, add 'USE_CUDA_GRAPHS': True to your hyperparameters.")


if __name__ == "__main__":
    main()
