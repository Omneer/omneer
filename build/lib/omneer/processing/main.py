#!/usr/bin/env python3
import shutil
import sys
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from omneer.processing.preprocess.preprocess import Data
from omneer.processing.misc import get_metrics, get_calibration_curve, pr_auc_score, compute_ci
from omneer.model.train import train
from omneer.processing.bootstrap import bootstrap
from omneer.visualization.plot import plot_roc, plot_pr
from omneer.processing.preprocess.features import select_top_features
from sklearn.metrics import confusion_matrix
from pathlib import Path

def main(csvfile, model_name, num_features=None):
    assert model_name in ['mlp', 'xgb', 'rf', 'lr', 'svm', 'lda', 'ensemble']

    # Directory to save results
    comb_dir = Path('./test')
    dset_dir = Path(csvfile).stem
    mode_dir = model_name
    save_dir = comb_dir / dset_dir / mode_dir

    # Clean up the result folder
    shutil.rmtree(save_dir, ignore_errors=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / 'checkpoints').mkdir(exist_ok=True)
    (save_dir / 'results').mkdir(exist_ok=True)

    # Initialize dataset
    csv_path = Path(__file__).resolve().parent.parent.joinpath('data', 'raw', csvfile)
    df = pd.read_csv(csv_path, encoding='latin1')

    if num_features:
        # Use the select_top_features function to get the selected features
        selected_features = select_top_features("PD", df.columns[2:], csv_path, num_features)
    else:
        selected_features = df.iloc[:, 2:]  # Use all features

    whole_data = Data(
        label='PD',
        features=selected_features,
        csv_dir=csv_path,
    )

    # For random data split
    train_size = int(len(whole_data) * 0.6)
    valid_size = len(whole_data) - train_size

    # Run for multiple times to collect the statistics of the performance metrics
    bootstrap(
        func=train,
        args=(model_name, whole_data, train_size, valid_size, str(save_dir)),
        kwargs={},
        num_runs=100,
        num_jobs=1,
    )

    # Calculate and show the mean & std of the metrics for all runs
    list_csv = (save_dir / 'results').glob('*')
    list_csv = [save_dir / 'results' / Path(fn).name for fn in list_csv]

    y_true_all, y_pred_all, scores_all = [], [], []
    for fn in list_csv:
        df = pd.read_csv(fn)
        y_true_all.append(df['Y Label'].to_numpy())
        y_pred_all.append(df['Y Predicted'].to_numpy())
        scores_all.append(df['Predicted score'].to_numpy())

    met_all = get_metrics(y_true_all, y_pred_all, scores_all)
    for k, v in met_all.items():
        if k not in ['Confusion Matrix']:
            lower, upper = compute_ci(v)  # Compute confidence interval
            print('{}:\t{:.4f} \u00B1 {:.4f}'.format(k, v[0], v[1], lower, upper).expandtabs(20))

    # Plot ROC, PR curves
    fig = plt.figure(figsize=(6, 6), dpi=100)
    plot_roc(plt.gca(), y_true_all, scores_all)
    fig.savefig(save_dir / f'ROC {model_name}.png', dpi=300)

    fig = plt.figure(figsize=(6, 6), dpi=100)
    plot_pr(plt.gca(), y_true_all, scores_all)
    fig.savefig(save_dir / f'PR {model_name}.png', dpi=300)

def cli():
    parser = argparse.ArgumentParser(description='Omneer command line interface.')
    parser.add_argument('csvfile', help='CSV file to process.')
    parser.add_argument('model', help='Model to use for data processing.')
    parser.add_argument('num_features', nargs='?', type=int, help='Number of features to use.')
    args = parser.parse_args()

    main(args.csvfile, args.model, args.num_features)

if __name__ == "__main__":
    cli()