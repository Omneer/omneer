#!/usr/bin/env python3

# Standard library imports
import argparse
import shutil
from pathlib import Path
import time

# Third-party imports
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# Local application imports
from omneer_cli.model.train import train
from omneer_cli.processing.bootstrap import bootstrap
from omneer_cli.processing.misc import get_metrics, compute_ci
from omneer_cli.processing.preprocess.features import select_top_features
from omneer_cli.processing.preprocess.preprocess import Data, preprocess_data
from omneer.visualization.plot import plot_roc, plot_pr

def predict(csvfile, model_name, num_features=None):
    """Main function to process data and run the model.

    Args:
        csvfile (str): Path to the CSV file to process.
        model_name (str): Name of the model to use.
        num_features (int, optional): Number of features to use. If None, all features are used.
    """
    assert model_name in ['mlp', 'xgb', 'ensemble', 'rf', 'lr', 'svm', 'lda', 'lgbm', 'tabnet', 'catboost']

    dset_dir = Path(csvfile).stem
    mode_dir = model_name
    home_dir = Path.home()
    omneer_files_dir = home_dir / "omneer_files"
    data_dir = omneer_files_dir / "data"
    save_dir = data_dir / dset_dir / mode_dir

    print(f"Debug: save_dir is {save_dir}")  # add this line for debugging

    # Clean up the result folder
    print(f"Debug: Removing directory {save_dir}")
    shutil.rmtree(save_dir, ignore_errors=True)
    print(f"Debug: Creating directory {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Debug: Creating directory {save_dir / 'checkpoints'}")
    (save_dir / 'checkpoints').mkdir(exist_ok=True)
    print(f"Debug: Creating directory {save_dir / 'results'}")
    (save_dir / 'results').mkdir(exist_ok=True)

    # Measure preprocessing time
    preprocessing_start_time = time.time()

    # Initialize dataset
    csv_path_raw = Path(csvfile)
    print(f"Debug: Reading file {csv_path_raw}")

    omneer_files_dir = home_dir / "omneer_files"
    data_dir = omneer_files_dir / "data"
    preprocessed_dir = data_dir / "preprocessing"
    csv_path_processed = preprocessed_dir / f'{dset_dir}_Preprocessed.csv'
    print(f"Debug: Reading file {csv_path_processed}")

    if not csv_path_processed.is_file():
        df = pd.read_csv(csv_path_raw, encoding='latin1')
        label_name = df.columns[1]  # Second column is the label
        features = df.columns[2:].tolist()  # Use all features

        if num_features:
            selected_features = select_top_features("PD", features, csv_path_raw, num_features)
        else:
            selected_features = features  # Use all features

        csv_path_processed = preprocess_data(csv_path_raw, label_name, len(selected_features), home_dir)

    df = pd.read_csv(csv_path_processed, encoding='latin1')
    selected_features = df.iloc[:, 2:]  # Use all features

    whole_data = Data(
        label='PD',
        features=selected_features,
        csv_dir=csv_path_processed,
        home_dir=home_dir,  # pass home_dir to Data class
        impute_method='iterative',
        scale_method='quantile'
    )

    # For random data split
    train_size = int(len(whole_data) * 0.6)
    valid_size = len(whole_data) - train_size

    preprocessing_end_time = time.time()
    preprocessing_elapsed_time = preprocessing_end_time - preprocessing_start_time
    print(f"Preprocessing completed in {preprocessing_elapsed_time:.2f} seconds.")

    # Run for multiple times to collect the statistics of the performance metrics
    bootstrap(
        func=train,
        args=(model_name, whole_data, train_size, valid_size, str(save_dir)),
        kwargs={},
        num_runs=100,
        num_jobs=1,
    )

    # Calculate and show the mean & std of the metrics for all runs
    list_csv = list((save_dir / 'results').glob('*'))

    y_true_all, y_pred_all, scores_all = [], [], []
    for fn in tqdm(list_csv, desc='Processing data', unit='file'):
        df = pd.read_csv(fn)
        y_true_all.append(df['Y Label'].to_numpy())
        y_pred_all.append(df['Y Predicted'].to_numpy())
        scores_all.append(df['Predicted score'].to_numpy())

    met_all = get_metrics(y_true_all, y_pred_all, scores_all)
    for k, v in met_all.items():
        if k not in ['Confusion Matrix']:
            lower, upper = compute_ci(v)
            print(f'{k:<20}{v[0]:.4f} ± {v[1]:.4f}')

    # Plot ROC, PR curves
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    plot_roc(ax, y_true_all, scores_all)
    fig.savefig(save_dir / f'ROC_{model_name}.png', dpi=300)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    plot_pr(ax, y_true_all, scores_all)
    fig.savefig(save_dir / f'PR_{model_name}.png', dpi=300)
