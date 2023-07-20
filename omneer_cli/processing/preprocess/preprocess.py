import torch
import pandas as pd
import numpy as np
import logging
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from sklearn.preprocessing import QuantileTransformer, RobustScaler, PowerTransformer, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LinearRegression
from pathlib import Path
from tqdm import tqdm
from tpot import TPOTClassifier
from imblearn.over_sampling import SMOTE
from joblib import Parallel, delayed
from sklearn.exceptions import NotFittedError

# Enable logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

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

# AutoML for feature selection
# AutoML for feature selection
# AutoML for feature selection
def automl_feature_selection(X, y, num_features):
    tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2)
    tpot.fit(X, y)

    # Get the feature importances of the final model
    feature_importances = tpot.fitted_pipeline_.steps[-1][1].feature_importances_

    # Get the indices of the features sorted by importance
    indices = np.argsort(feature_importances)

    # Select the top num_features features
    top_indices = indices[-num_features:]

    # Return only the top num_features features
    return X[:, top_indices]


# Data augmentation using SMOTE
def smote_augmentation(X, y):
    smote = SMOTE()
    return smote.fit_resample(X, y)

# Advanced scaling method
def yeo_johnson_scaling(X):
    scaler = PowerTransformer(method='yeo-johnson')
    return scaler.fit_transform(X)

class Data(torch.utils.data.Dataset):
    def __init__(self, label, features, csv_dir, home_dir, 
                 impute_method='iterative', scale_method='quantile', 
                 outlier_detection=False, feature_selection=None, transform_method=None, 
                 augment_data=False, handle_categorical='onehot', num_features=None):
        self.features = features
        self.label = label
        self.home_dir = home_dir
        self.impute_method = impute_method
        self.scale_method = scale_method
        self.outlier_detection = outlier_detection
        self.feature_selection = feature_selection
        self.transform_method = transform_method
        self.augment_data = augment_data
        self.handle_categorical = handle_categorical
        content = self.read_csv(csv_dir)
        self.content = content.dropna(subset=self.features)  # Remove rows with missing features
        self.content.dropna(subset=[self.label], inplace=True)  # Remove rows with missing labels
        self.x, self.y = self.process_data()

    def read_csv(self, csv_file):
        return pd.read_csv(csv_file)

    def filter_incomplete_cases(self, df):
        return df.dropna(subset=[self.label] + self.features)

    def read_csv(self, csv_file):
        return pd.read_csv(csv_file)

    def filter_incomplete_cases(self, df):
        return df.dropna(subset=self.features + [self.label])

    def process_data(self):
        # Separate features into numerical and categorical
        numerical_features = self.content[self.features].select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = self.content[self.features].select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        # Define the preprocessing for numerical and categorical features
        numerical_transformer = Pipeline(steps=[
            ('imputer', CustomImputer(method=self.impute_method)),
            ('scaler', PowerTransformer(method='yeo-johnson'))])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        # Combine the transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)])

        # Fit and transform the features
        X = preprocessor.fit_transform(self.content[self.features])

        # Apply feature selection using AutoML
        if self.feature_selection == 'automl':
            X = automl_feature_selection(X, self.content[self.label])

        # Apply SMOTE data augmentation
        if self.augment_data:
            X, y = smote_augmentation(X, self.content[self.label])

        # Outlier detection
        if self.outlier_detection:
            clf = IsolationForest(contamination=0.1)
            outliers = clf.fit_predict(X)
            X = X[outliers == 1]
            y = y[outliers == 1]

        return X, y

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
                    outlier_detection=False, feature_selection=None, transform_method=None,
                    augment_data=False, handle_categorical='onehot'):
    df = pd.read_csv(file_path, encoding='latin1')

    # Extract all the columns. The label column is the second column in the data.
    all_cols = df.columns.tolist()

    # But remove the patient and label columns from the features we want to preprocess
    features = all_cols[2:feature_count+2]

    data_preprocess = Data(
        label=label_name,
        features=features,
        csv_dir=file_path,
        home_dir=home_dir,
        impute_method=impute_method,
        scale_method=scale_method,
        outlier_detection=outlier_detection,
        feature_selection=feature_selection,
        transform_method=transform_method,
        augment_data=augment_data,
        handle_categorical=handle_categorical
    )

    # Prepare a DataFrame
    x, y = data_preprocess.all
    df_preprocessed = pd.DataFrame(x)

    # Add the label as the first column
    df_preprocessed.insert(0, label_name, y)

    # If PCA or linear feature selection was applied, rename the transformed features
    if data_preprocess.feature_selection == 'automl':
        df_preprocessed.columns = [label_name] + [f'automl_feature_{i}' for i in range(df_preprocessed.shape[1]-1)]
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
                scale_method='yeo-johnson',
                outlier_detection=False,
                feature_selection='automl',
                transform_method=None,
                augment_data=True,
                handle_categorical='onehot'
            )