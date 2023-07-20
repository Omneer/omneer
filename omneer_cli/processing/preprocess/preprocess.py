import torch
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from sklearn.preprocessing import QuantileTransformer, RobustScaler, PowerTransformer, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LinearRegression
from pathlib import Path
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
    def __init__(self, label, features, csv_dir, home_dir, 
                 impute_method='iterative', scale_method='quantile', 
                 outlier_detection=False, feature_selection=None, transform_method=None):
        self.features = features
        self.label = label
        self.home_dir = home_dir
        self.impute_method = impute_method
        self.scale_method = scale_method
        self.outlier_detection = outlier_detection
        self.feature_selection = feature_selection
        self.transform_method = transform_method
        content = self.read_csv(csv_dir)
        self.content = self.filter_incomplete_cases(content)
        self.x, self.y = self.process_data()

    def read_csv(self, csv_file):
        return pd.read_csv(csv_file)

    def filter_incomplete_cases(self, df):
        return df.dropna(subset=self.features + [self.label])

    def process_data(self):
        x = self.content[self.features].values
        y = self.content[self.label].values

        # Create preprocessing pipeline
        preprocessing_steps = [('imputer', CustomImputer(method=self.impute_method))]

        if self.transform_method == 'log':
            preprocessing_steps.append(('transformer', FunctionTransformer(np.log1p)))
        elif self.transform_method == 'sqrt':
            preprocessing_steps.append(('transformer', FunctionTransformer(np.sqrt)))

        if self.scale_method == 'robust':
            preprocessing_steps.append(('scaler', RobustScaler()))
        elif self.scale_method == 'quantile':
            preprocessing_steps.append(('scaler', QuantileTransformer()))
        elif self.scale_method == 'standard':
            preprocessing_steps.append(('scaler', StandardScaler()))
        elif self.scale_method == 'minmax':
            preprocessing_steps.append(('scaler', MinMaxScaler()))

        if self.feature_selection == 'pca':
            preprocessing_steps.append(('selector', PCA(n_components=0.95)))
        elif self.feature_selection == 'linear':
            preprocessing_steps.append(('selector', SelectFromModel(LinearRegression())))

        preprocessing_pipeline = Pipeline(steps=preprocessing_steps)
        x = preprocessing_pipeline.fit_transform(x, y)

        if self.outlier_detection:
            clf = IsolationForest(contamination=0.1)
            outliers = clf.fit_predict(x)
            x = x[outliers == 1]
            y = y[outliers == 1]

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

def preprocess_data(file_path, label_name, feature_count, home_dir, 
                    impute_method='iterative', scale_method='quantile', 
                    outlier_detection=False, feature_selection=None, transform_method=None):
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
        impute_method=impute_method,
        scale_method=scale_method,
        outlier_detection=outlier_detection,
        feature_selection=feature_selection,
        transform_method=transform_method
    )

    # Prepare a DataFrame
    x, y = data_preprocess.all
    df_preprocessed = pd.DataFrame(x)

    # Add the label as the first column
    df_preprocessed.insert(0, label_name, y)

    # If PCA or linear feature selection was applied, rename the transformed features
    if data_preprocess.feature_selection == 'pca':
        df_preprocessed.columns = [label_name] + [f'PCA_feature_{i}' for i in range(df_preprocessed.shape[1]-1)]
    elif data_preprocess.feature_selection == 'linear':
        df_preprocessed.columns = [label_name] + [f'linear_feature_{i}' for i in range(df_preprocessed.shape[1]-1)]
    else:
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

            preprocess_data(
                file_path=file_path,
                label_name=label,
                feature_count=len(features),
                home_dir=home_dir,
                impute_method='iterative',
                scale_method='quantile',
                outlier_detection=False,
                feature_selection=None,
                transform_method=None
            )
