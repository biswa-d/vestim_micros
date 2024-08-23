#----------------------------------------------------------------------------------------
# Description: This script contains the backend code for training a LSTM model in PyTorch for certain number of epochs and repetitions 
# based on the hyper parameters entered by the user in the GUI interface.
#
# Created on: Wed Aug 07 2024 15:59:50
# Author: Biswanath Dehury
# Company: Dr. Phil Kollmeyer's Battery Lab at McMaster University
# Copyright (c) 2024 Biswanath Dehury, Dr. Phil Kollmeyer's Battery Lab at McMaster University
#----------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------
#Descrition: Updatated Training script to handle model training on the backend with provisions for randomisation and logging
#
# Created On: Thu Aug 08 2024 16:55:35
# Author: Biswanath Dehury
# Company: Dr. Phil Kollmeyer's Battery Lab at McMaster University
# Copyright (c) 2024 Biswanath Dehury, Dr. Phil Kollmeyer's Battery Lab at McMaster University
#----------------------------------------------------------------------------------------


import torch
import copy
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split, SubsetRandomSampler
import datetime
import time

# Define the seed based on the current timestamp
SEED = int(time.time())  # Time-based seed for randomness

# Function to create dataset sequences
def create_dataset(X_data, Y_data, lookback):
    X, y = [], []
    for i in range(lookback, len(Y_data)):
        X.append(X_data[i - lookback:i])
        y.append(Y_data[i])
    return np.array(X), np.array(y)

# Function to load and process CSV files into data sequences
def load_and_process_data(folder_path, lookback):
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    data_sequences = []
    target_sequences = []

    for file in csv_files:
        df = pd.read_csv(file)
        X_data = df[['Voltage', 'Current', 'Temp']].values
        Y_data = df['SOC'].values
        X, y = create_dataset(X_data, Y_data, lookback)
        data_sequences.append(X)
        target_sequences.append(y)

    X_combined = np.concatenate(data_sequences, axis=0)
    y_combined = np.concatenate(target_sequences, axis=0)
    
    # Deleting original cache after processing
    del data_sequences, target_sequences

    return X_combined, y_combined

# Function to create DataLoaders with SubsetRandomSampler
def create_data_loaders(X, y, batch_size, train_split=0.7, seed=SEED):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    train_size = int(dataset_size * train_split)
    valid_size = dataset_size - train_size

    np.random.seed(seed)
    np.random.shuffle(indices)

    train_indices, valid_indices = indices[:train_size], indices[train_size:]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True)
    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, drop_last=True)

    # Clean up cache variables after DataLoaders are created
    del X_tensor, y_tensor, indices, train_indices, valid_indices

    return train_loader, valid_loader

# Training function with logging and sending data to GUI
def train(model, params, update_progress):
    lookback = params.get('LOOKBACK', 400)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    train_folder = params['TRAIN_FOLDER']
    X_data, Y_data = load_and_process_data(train_folder, lookback)

    best_model = None
    best_validation_error = float('inf')
    patience_counter = 0
    train_loss, valid_loss = [], []

    for repetition in range(params['REPETITIONS']):
        print(f"Starting repetition {repetition + 1}/{params['REPETITIONS']}")
        seed = SEED + repetition

        train_loader, valid_loader = create_data_loaders(X_data, Y_data, params['BATCH_SIZE'], seed=seed)

        model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['INITIAL_LR'])
        criterion = torch.nn.MSELoss()

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[params['MAX_EPOCHS'] // 10 * i for i in range(2, 10, 2)],
            gamma=0.1
        )

        for epoch in range(1, params['MAX_EPOCHS'] + 1):
            total_train_loss = []
            model.train()
            for X_batch, y_batch in train_loader:
                h_s = torch.zeros(params['LAYERS'], params['BATCH_SIZE'], params['HIDDEN_UNITS']).to(device)
                h_c = torch.zeros(params['LAYERS'], params['BATCH_SIZE'], params['HIDDEN_UNITS']).to(device)
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                # Forward pass
                y_pred, (h_s, h_c) = model(X_batch, h_s, h_c)
                y_pred = y_pred.squeeze(-1)  # Ensure y_pred is 2D [batch_size, 1] -> [batch_size]

                loss = criterion(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_loss.append(loss.item())
            
            train_loss_avg = np.mean(total_train_loss)
            train_loss.append(train_loss_avg)
            scheduler.step()

            if epoch == 1 or epoch % params['ValidFrequency'] == 0 or epoch == params['MAX_EPOCHS']:
                validation_error = validate_model(model, params, valid_loader, criterion, device)
                valid_loss.append(validation_error)
                
                # Send data to the GUI via update_progress, including repetition number
                update_progress(repetition + 1, epoch, train_loss, valid_loss)

                if validation_error < best_validation_error:
                    best_validation_error = validation_error
                    best_model = copy.deepcopy(model)
                    torch.save({'epoch': epoch, 'model': best_model}, './best_model.pth')
                    torch.save(optimizer.state_dict(), './optimizer.pth')
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter > params['VALID_PATIENCE']:
                    print(f"Early stopping at epoch {epoch + 1} due to no improvement.")
                    break

    return best_model

# Validation function
def validate_model(model, params, valid_loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for X_batch, y_batch in valid_loader:
            h_s = torch.zeros(params['LAYERS'], params['BATCH_SIZE'], params['HIDDEN_UNITS']).to(device)
            h_c = torch.zeros(params['LAYERS'], params['BATCH_SIZE'], params['HIDDEN_UNITS']).to(device)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            y_pred, (h_s, h_c) = model(X_batch, h_s, h_c)
            y_pred = y_pred.squeeze(-1)  # Ensure y_pred is 2D [batch_size, 1] -> [batch_size]

            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()

    average_loss = total_loss / len(valid_loader)
    return average_loss

# Logging function
def log(filename, log_string):
    with open(filename, 'a') as f:
        f.write(f"{datetime.datetime.now()} - {log_string}\n")
