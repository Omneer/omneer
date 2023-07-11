import torch
import csv
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import QuantileTransformer, RobustScaler, PowerTransformer, StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import Lasso
from pathlib import Path
import os
from tqdm import tqdm

# Custom Transformer for handling missing values
class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, method='iterative', n_neighbors=5, max_iter=10):
        self.method = method
        self.n_neighbors = n_neighbors
        self.max_iter = max_iter

    def fit(self, X, y=None):
        if self.method == 'iterative':
            self.imputer_ = IterativeImputer(max_iter=self.max_iter, random_state=0)
        elif self.method == 'knn':
            self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors)
        else:
            self.imputer_ = SimpleImputer(strategy=self.method)
        self.imputer_.fit(X)
        return self

    def transform(self, X):
        return self.imputer_.transform(X)

class Data(torch.utils.data.Dataset):
    def __init__(self, label, features, csv_dir, home_dir, impute_method='iterative', scale_method='quantile'):
        self.features = features
        self.label = label
        self.home_dir = home_dir
        self.impute_method = impute_method
        self.scale_method = scale_method
        content = self.read_csv(csv_dir)
        self.content = self.filter_incomplete_cases(content)
        self.x, self.y = self.process_data()

    def read_csv(self, csv_file):
        content = []
        with open(csv_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                content.append(row)
        return content

    def filter_incomplete_cases(self, content):
        # Identify empty columns
        empty_columns = []
        for key in self.features:
            column_values = [row.get(key, '') for row in content]
            if all(value == '' for value in column_values):
                empty_columns.append(key)

        # Remove empty columns
        self.features = [key for key in self.features if key not in empty_columns]

        # Remove rows with empty values in any remaining features or label
        filtered_content = []
        for row in content:
            if all(row.get(key, '') != '' for key in self.features) and row.get(self.label, '') != '':
                filtered_row = {key: row[key] for key in self.features}
                filtered_row[self.label] = row[self.label]
                filtered_content.append(filtered_row)

        return filtered_content

    def process_data(self):
        x = []
        y = []
        for row in self.content:
            x_row = []
            for key in self.features:
                x_row.append(float(row[key]))
            x.append(x_row)
            y.append(float(row[self.label]))
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        # Create preprocessing pipeline
        preprocessing_steps = [('imputer', CustomImputer(method=self.impute_method))]
        if self.scale_method == 'imputer':
            preprocessing_steps.append(('scaler', QuantileTransformer()))
        elif self.scale_method == 'quantile':
            preprocessing_steps.append(('scaler', StandardScaler()))
        elif self.scale_method == 'minmax':
            preprocessing_steps.append(('scaler', MinMaxScaler()))
        preprocessing_pipeline = Pipeline(steps=preprocessing_steps)
        x = preprocessing_pipeline.fit_transform(x)

        return x, y

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def input_length(self):
        return len(self.features)

    @property
    def all(self):
        return self.x, self.y

    def save_preprocessed_data(self, file_name):
        # Prepare a DataFrame
        df = pd.DataFrame(self.x)
        df.insert(0, self.label, self.y)
        df.columns = [self.label] + self.features

        # Extract the file name without the extension
        file_name_without_extension = Path(file_name).stem

        # Set the directory path for saving the preprocessed file
        output_dir = self.home_dir / 'omneer_files' / 'data' / 'preprocessed'

        # Create the directory if it does not exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save the DataFrame to a csv file
        output_path = output_dir / f"{file_name_without_extension}_preprocessed.csv"
        df.to_csv(output_path, index=False)

def preprocess_data(file_path, label_name, feature_count, home_dir):
    df = pd.read_csv(file_path, encoding='latin1')

    # Extract all the columns except 'Patient'. The 'Patient' column should be the first column in the data.
    all_cols = df.columns.tolist()

    # But remove it from the features we want to preprocess
    all_cols.remove('Patient')
    features = all_cols[:feature_count]

    # Ensure label is not also treated as a feature
    features.remove(label_name)

    data_preprocess = Data(
        label=label_name,
        features=features,
        csv_dir=file_path,
        home_dir=home_dir,
        impute_method='iterative',
        scale_method='quantile'
    )

    # Prepare a DataFrame
    x, y = data_preprocess.all
    df_preprocessed = pd.DataFrame(x)

    # Add the label as the first column
    df_preprocessed.insert(0, label_name, y)
    df_preprocessed.columns = [label_name] + features

    # Add back the 'Patient' column to the first position of the preprocessed dataframe
    df_preprocessed.insert(0, 'Patient', df['Patient'])

    # Extract the stub of the file path without the extension
    file_name_without_extension = Path(file_path).stem

    # Save the DataFrame to a csv file
    output_path = home_dir / 'omneer_files' / 'data' / 'preprocessed' / f"{file_name_without_extension}_preprocessed.csv"
    df_preprocessed.to_csv(output_path, index=False)

    return output_path

if __name__ == "__main__":
    home_dir = Path.home()
    csv_dir = home_dir / 'omneer_files' / 'data' / 'raw'
    for file_path in csv_dir.iterdir():
        if file_path.name.endswith('.csv'):

            df = pd.read_csv(file_path, encoding='latin1')

            patient = 'Patient'
            label = 'PD'
            features = df.columns[2:].tolist()

            data = Data(
                label=label,
                features=features,
                csv_dir=file_path,
                home_dir=home_dir,  # pass home_dir to Data class
                impute_method='iterative',
                scale_method='quantile'
            )

            data.save_preprocessed_data(file_path)