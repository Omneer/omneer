#!/usr/bin/env python3
import os
import shutil
import sys
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from omneer.data.preprocessing.preprocess import Data
from omneer.processing.misc import get_metrics, get_calibration_curve, pr_auc_score, compute_ci
from omneer.model.train import train
from omneer.processing.bootstrap import bootstrap
from omneer.visualization.plot import plot_roc, plot_pr
from sklearn.metrics import confusion_matrix


def main(csvfile, model_name):
    assert model_name in ['mlp', 'xgb', 'rf', 'lr', 'svm', 'lda', 'ensemble']

    # Directory to save results
    comb_dir = './test'
    dset_dir = os.path.splitext(os.path.basename(csvfile))[0]
    mode_dir = model_name
    save_dir = '{}/{}/{}'.format(comb_dir, dset_dir, mode_dir)

    # Clean up the result folder
    shutil.rmtree(save_dir, ignore_errors=True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir + '/checkpoints', exist_ok=True)
    os.makedirs(save_dir + '/results', exist_ok=True)

    # Initialize dataset
    df = pd.read_csv(csvfile, encoding='latin1')
    whole_data = Data(
        label='PD',
        features=df.iloc[:, 1:],
        csv_dir=csvfile,
    )

    # For random data split
    train_size = int(len(whole_data) * 0.6)
    valid_size = len(whole_data) - train_size

    # Run for multiple times to collect the statistics of the performance metrics
    bootstrap(
        func=train,
        args=(model_name, whole_data, train_size, valid_size, save_dir),
        kwargs={},
        num_runs=100,
        num_jobs=1,
    )

    # Calculate and show the mean & std of the metrics for all runs
    list_csv = os.listdir(save_dir + '/results')
    list_csv = ['{}/results/{}'.format(save_dir, fn) for fn in list_csv]

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
    fig.savefig('{}/ROC {}.png'.format(save_dir, model_name), dpi=300)

    fig = plt.figure(figsize=(6, 6), dpi=100)
    plot_pr(plt.gca(), y_true_all, scores_all)
    fig.savefig('{}/PR {}.png'.format(save_dir, model_name), dpi=300)


def cli():
    parser = argparse.ArgumentParser(description='Omneer command line interface.')
    parser.add_argument('csvfile', help='CSV file to process.')
    parser.add_argument('model', help='Model to use for data processing.')
    args = parser.parse_args()

    main(args.csvfile, args.model)

if __name__ == "__main__":
    cli()
