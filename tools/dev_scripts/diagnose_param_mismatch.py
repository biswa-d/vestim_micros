"""Diagnose parameter-count mismatches between training and testing summaries.

Usage:
  python vestim\scripts\diagnose_param_mismatch.py --job-dir <job_folder>
  python vestim\scripts\diagnose_param_mismatch.py --progress-csv <path/to/training_progress.csv>

The script loads `job_metadata.json` and the training progress CSV if available,
then prints recorded hyperparams and recomputes expected parameter counts
for LSTM/GRU and FNN using the same formulas as the GUI.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import csv
import sys

import numpy as np


def load_job_metadata(job_dir: Path):
    meta = {}
    meta_path = job_dir / 'job_metadata.json'
    if meta_path.exists():
        try:
            with meta_path.open('r', encoding='utf-8') as f:
                meta = json.load(f)
        except Exception as e:
            print(f"Failed to read job_metadata.json: {e}")
    return meta


def parse_training_csv(csv_path: Path):
    if not csv_path.exists():
        return None
    # Read header only
    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def calc_fnn_params(input_size, hidden_layer_sizes, output_size=1):
    total = 0
    prev = input_size
    for h in hidden_layer_sizes:
        total += prev * h + h
        prev = h
    total += prev * output_size + output_size
    return total


def calc_rnn_params(layer_sizes, input_size, gates=4):
    # layer_sizes: list of ints per layer
    total = 0
    # first layer
    first = layer_sizes[0]
    total += gates * (input_size + first) * first + gates * first
    # subsequent
    for i in range(1, len(layer_sizes)):
        prev = layer_sizes[i-1]
        cur = layer_sizes[i]
        total += gates * (prev + cur) * cur + gates * cur
    # output
    total += layer_sizes[-1] * 1 + 1
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', help='Path to job folder containing job_metadata.json')
    parser.add_argument('--progress-csv', help='Path to training_progress.csv')
    args = parser.parse_args()

    job_dir = None
    if args.job_dir:
        job_dir = Path(args.job_dir)
        if not job_dir.exists():
            print('job-dir does not exist')
            sys.exit(1)

    csv_rows = None
    if args.progress_csv:
        csv_path = Path(args.progress_csv)
        if not csv_path.exists():
            print('progress csv does not exist')
            sys.exit(1)
        csv_rows = parse_training_csv(csv_path)
        # try to locate job_dir from csv if not provided
        if job_dir is None:
            # walk up parents to find job_metadata
            for p in [csv_path] + list(csv_path.parents):
                if (Path(p) / 'job_metadata.json').exists():
                    job_dir = Path(p)
                    break

    if job_dir is None:
        print('Could not determine job folder. Provide --job-dir or run next to training_progress.csv.')
        sys.exit(1)

    meta = load_job_metadata(job_dir)
    print(f"Job folder: {job_dir}")
    print("job_metadata.json contents (filtered):")
    for k in ['normalized_columns','scaler_path','normalization_applied','data_loader_params','target_column']:
        if k in meta:
            print(f"  {k}: {meta[k]}")

    # If hyperparams stored in job metadata
    hyper = meta.get('hyperparams') or {}
    if not hyper:
        # try reading saved hyperparams file
        hp_path = job_dir / 'hyperparams.json'
        if hp_path.exists():
            try:
                with hp_path.open('r', encoding='utf-8') as f:
                    hyper = json.load(f)
            except Exception:
                pass

    print('\nHyperparameters from metadata or hyperparams.json (selected):')
    keys = ['INPUT_SIZE','HIDDEN_UNITS','LAYERS','RNN_LAYER_SIZES','HIDDEN_LAYER_SIZES','NUM_LEARNABLE_PARAMS','FEATURE_COLUMNS']
    for k in keys:
        if k in hyper:
            print(f"  {k}: {hyper[k]}")

    # Also show model_metadata written by training setup
    task_model_meta = meta.get('model_metadata') or {}
    if task_model_meta:
        print('\nmodel_metadata (job folder):')
        for k in ['input_size','num_learnable_params','model_type']:
            if k in task_model_meta:
                print(f"  {k}: {task_model_meta[k]}")

    # Recompute counts if possible
    input_size = None
    if 'INPUT_SIZE' in hyper:
        try:
            input_size = int(hyper['INPUT_SIZE'])
        except Exception:
            pass
    # fallback to model_metadata input_size
    if input_size is None and 'input_size' in task_model_meta:
        input_size = int(task_model_meta['input_size'])

    # Get target column and feature columns
    feature_columns = None
    if 'FEATURE_COLUMNS' in hyper:
        feature_columns = hyper['FEATURE_COLUMNS']
        try:
            if isinstance(feature_columns, str):
                # try to parse comma-separated
                feature_columns = [c.strip() for c in feature_columns.split(',') if c.strip()]
        except Exception:
            pass
    if feature_columns is not None:
        print(f"Detected FEATURE_COLUMNS length: {len(feature_columns)}")
        if input_size is None:
            input_size = len(feature_columns)

    print(f"Using input_size={input_size}")

    # Get RNN layer sizes
    rnn_layer_sizes = None
    if 'RNN_LAYER_SIZES' in hyper and hyper['RNN_LAYER_SIZES']:
        val = hyper['RNN_LAYER_SIZES']
        if isinstance(val, str):
            rnn_layer_sizes = [int(x.strip()) for x in val.split(',')]
        elif isinstance(val, list):
            rnn_layer_sizes = [int(x) for x in val]
    elif 'HIDDEN_UNITS' in hyper:
        try:
            hidden = int(hyper['HIDDEN_UNITS'])
            layers = int(hyper.get('LAYERS', 1))
            rnn_layer_sizes = [hidden] * layers
        except Exception:
            pass

    if rnn_layer_sizes is not None and input_size is not None:
        lstm_count = calc_rnn_params(rnn_layer_sizes, input_size, gates=4)
        print(f"Recomputed LSTM params (using layer sizes {rnn_layer_sizes}, input_size={input_size}): {lstm_count}")
        recorded = hyper.get('NUM_LEARNABLE_PARAMS') or task_model_meta.get('num_learnable_params')
        print(f"Recorded NUM_LEARNABLE_PARAMS: {recorded}")

    # FNN
    if 'HIDDEN_LAYER_SIZES' in hyper and input_size is not None:
        hvals = hyper['HIDDEN_LAYER_SIZES']
        if isinstance(hvals, str):
            hlist = [int(x.strip()) for x in hvals.split(',')]
        elif isinstance(hvals, list):
            hlist = [int(x) for x in hvals]
        else:
            hlist = []
        fnn_count = calc_fnn_params(input_size, hlist, output_size=1)
        print(f"Recomputed FNN params (hidden {hlist}, input_size={input_size}): {fnn_count}")

    print('\nDone.')


if __name__ == '__main__':
    main()
