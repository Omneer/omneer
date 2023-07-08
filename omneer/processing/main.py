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
from omneer.model.train import train
from omneer.processing.bootstrap import bootstrap
from omneer.processing.misc import get_metrics, compute_ci
from omneer.processing.preprocess.features import select_top_features
from omneer.processing.preprocess.preprocess import Data
from omneer.visualization.plot import plot_roc, plot_pr


def main(csvfile, model_name, num_features=None):
    """Main function to process data and run the model.

    Args:
        csvfile (str): Path to the CSV file to process.
        model_name (str): Name of the model to use.
        num_features (int, optional): Number of features to use. If None, all features are used.
    """
    assert model_name in ['mlp', 'xgb', 'rf', 'lr', 'svm', 'lda', 'ensemble']

    # Set up directories
    comb_dir = Path('./test')
    dset_dir = Path(csvfile).stem
    mode_dir = model_name
    save_dir = comb_dir / dset_dir / mode_dir

    # Clean up the result folder
    shutil.rmtree(save_dir, ignore_errors=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / 'checkpoints').mkdir(exist_ok=True)
    (save_dir / 'results').mkdir(exist_ok=True)

    # Measure preprocessing time
    preprocessing_start_time = time.time()

    # Initialize dataset
    raw_dir = Path(__file__).resolve().parent.parent / 'data' / 'raw'
    csv_path_raw = raw_dir / csvfile
    processed_dir = Path(__file__).resolve().parent.parent / 'data' / 'preprocessing'
    csv_path_processed = processed_dir / f'{dset_dir}_Preprocessed.csv'

    if not csv_path_processed.is_file():
        df = pd.read_csv(csv_path_raw, encoding='latin1')
        features = df.columns[2:].tolist()  # Use all features

        data = Data(
            label='PD',
            features=features,
            csv_dir=csv_path_raw,
        )
        data.save_preprocessed_data(csv_path_processed)

    df = pd.read_csv(csv_path_processed, encoding='latin1')

    if num_features:
        selected_features = select_top_features("PD", df.columns[2:], csv_path_processed, num_features)
    else:
        selected_features = df.iloc[:, 2:]  # Use all features

    whole_data = Data(
        label='PD',
        features=selected_features,
        csv_dir=csv_path_processed,
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


def cli():
    """Command line interface for the script."""
    parser = argparse.ArgumentParser(description='Omneer command line interface.')
    parser.add_argument('csvfile', help='CSV file to process.')
    parser.add_argument('model', help='Model to use for data processing.')
    parser.add_argument('num_features', nargs='?', type=int, help='Number of features to use.')
    args = parser.parse_args()

    main(args.csvfile, args.model, args.num_features)


if __name__ == "__main__":
    cli()
