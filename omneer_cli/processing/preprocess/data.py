from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import torch
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path
from omneer_cli.processing.preprocess.feature_selection import permutation_importance_feature_selection
from omneer_cli.processing.preprocess.utils import CustomImputer, AutoencoderFeatureExtractor, advanced_scaling, advanced_categorical_encoding, advanced_feature_extraction, data_augmentation, advanced_outlier_detection, data_cleaning


class Data(torch.utils.data.Dataset):
    def __init__(self, label, features, csv_dir, home_dir, 
                 impute_method='iterative', scale_method='quantile', 
                 outlier_detection=False, feature_selection=None, transform_method=None, 
                 augment_data=False, handle_categorical='onehot', num_features=None,
                 data_augmentation_method='smote', outlier_detection_method='isolation_forest',
                 feature_extraction_method=None, categorical_encoding_method='onehot'):
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
        self.data_augmentation_method = data_augmentation_method
        self.outlier_detection_method = outlier_detection_method
        self.feature_extraction_method = feature_extraction_method
        self.categorical_encoding_method = categorical_encoding_method
        content = self.read_csv(csv_dir)
        self.content = content.dropna(subset=self.features)  # Remove rows with missing features
        self.content.dropna(subset=[self.label], inplace=True)  # Remove rows with missing labels
        self.x, self.y = self.process_data()

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
            ('scaler', FunctionTransformer(advanced_scaling, kw_args={'method': self.scale_method}))])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', FunctionTransformer(advanced_categorical_encoding, kw_args={'method': self.categorical_encoding_method}))])

        # Combine the transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)])

        # Fit and transform the features
        X = preprocessor.fit_transform(self.content[self.features])

        # Apply feature extraction
        if self.feature_extraction_method:
            X = advanced_feature_extraction(X, method=self.feature_extraction_method)

        # Apply feature selection using AutoML
        if self.feature_selection == 'automl':
            X = permutation_importance_feature_selection(X, self.content[self.label])
            model = LogisticRegression()
            score = np.mean(cross_val_score(model, X, self.content[self.label], cv=5))
            logging.info(f"Cross-validation score after feature selection: {score}")
        
        if self.feature_extraction_method == 'autoencoder':
            extractor = AutoencoderFeatureExtractor()
        else:
            X = advanced_feature_extraction(X, method=self.feature_extraction_method)


        # Apply data augmentation
        if self.augment_data:
            X, y = data_augmentation(X, self.content[self.label], method=self.data_augmentation_method)
        else:
            y = self.content[self.label]

        # Outlier detection
        if self.outlier_detection:
            X = advanced_outlier_detection(X, method=self.outlier_detection_method)

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