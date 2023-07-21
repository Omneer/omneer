import pandas as pd
import torch
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
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, RFE
from sklearn.preprocessing import FunctionTransformer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA
from umap import UMAP
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from category_encoders import TargetEncoder, CatBoostEncoder, BinaryEncoder, HashingEncoder
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from pathlib import Path
import joblib
import imblearn.under_sampling as undersampling
import imblearn.ensemble as ensemble_methods
from sklearn.inspection import permutation_importance
from tpot import TPOTClassifier
import h2o
from h2o.automl import H2OAutoML
from sklearn.feature_selection import VarianceThreshold
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, load_model
#from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
#from pytorch_tabnet.tab_model import TabNetClassifier

# Enable logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

# Custom Transformer for handling missing values
# Advanced Imputation Techniques
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
        elif self.method == 'simple':
            self.imputer_ = SimpleImputer(strategy='mean')
        self.imputer_.fit(X)
        return self

    def transform(self, X):
        return self.imputer_.transform(X)
    
# Deep Learning-based Feature Extraction
class AutoencoderFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, encoding_dim=32, epochs=50):
        self.encoding_dim = encoding_dim
        self.epochs = epochs

    def fit(self, X, y=None):
        input_dim = X.shape[1]
        
        input_layer = Input(shape=(input_dim, ))
        encoder_layer = Dense(self.encoding_dim, activation="relu")(input_layer)
        decoder_layer = Dense(input_dim, activation='sigmoid')(encoder_layer)
        
        self.autoencoder = Model(inputs=input_layer, outputs=decoder_layer)
        self.encoder = Model(inputs=input_layer, outputs=encoder_layer)
        
        self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        self.autoencoder.fit(X, X, epochs=self.epochs, batch_size=256, shuffle=True, verbose=0)
        
        return self

    def transform(self, X):
        return self.encoder.predict(X)

# AutoML for feature selection
def permutation_importance_feature_selection(X, y, model=RandomForestClassifier(), num_features=None):
    model.fit(X, y)
    result = permutation_importance(model, X, y, n_repeats=10)
    sorted_idx = result.importances_mean.argsort()
    
    if num_features is not None:
        sorted_idx = sorted_idx[-num_features:]
    
    return X[:, sorted_idx]

# Data Cleaning
def data_cleaning(X):
    # Remove duplicate columns
    _, unique_columns = np.unique(X, axis=1, return_index=True)
    X = X[:, unique_columns]

    # Remove constant and quasi-constant columns
    constant_filter = VarianceThreshold(threshold=0.01)
    X = constant_filter.fit_transform(X)

    return X

# Data augmentation using various methods
def data_augmentation(X, y, method='smote'):
    if method == 'smote':
        augmenter = SMOTE()
    elif method == 'adasyn':
        augmenter = ADASYN()
    elif method == 'borderline_smote':
        augmenter = BorderlineSMOTE()
    elif method == 'svm_smote':
        augmenter = SVMSMOTE()
    X_resampled, y_resampled = augmenter.fit_resample(X, y)
    return X_resampled, y_resampled

# Advanced scaling method
def advanced_scaling(X, method='yeo_johnson'):
    if method == 'yeo_johnson':
        scaler = PowerTransformer(method='yeo-johnson')
    elif method == 'quantile_normal':
        scaler = QuantileTransformer(output_distribution='normal')
    elif method == 'quantile_uniform':
        scaler = QuantileTransformer(output_distribution='uniform')
    elif method == 'standard':
        scaler = StandardScaler()
    elif method == 'min_max':
        scaler = MinMaxScaler()
    return scaler.fit_transform(X)

# Advanced categorical encoding
def advanced_categorical_encoding(X, method='onehot'):
    if method == 'onehot':
        encoder = OneHotEncoder(handle_unknown='ignore')
    elif method == 'target':
        encoder = TargetEncoder()
    elif method == 'catboost':
        encoder = CatBoostEncoder()
    elif method == 'binary':
        encoder = BinaryEncoder()
    elif method == 'hashing':
        encoder = HashingEncoder()
    return encoder.fit_transform(X)

# Advanced outlier detection
def advanced_outlier_detection(X, method='isolation_forest'):
    if method == 'isolation_forest':
        detector = IsolationForest(contamination=0.1)
    elif method == 'local_outlier_factor':
        detector = LocalOutlierFactor(novelty=True)
    elif method == 'one_class_svm':
        detector = OneClassSVM(nu=0.1)
    outliers = detector.fit_predict(X)
    return X[outliers == 1]

# Advanced feature extraction
def advanced_feature_extraction(X, method='pca', n_components=2):
    if method == 'pca':
        extractor = PCA(n_components=n_components)
    elif method == 'kernel_pca':
        extractor = KernelPCA(n_components=n_components)
    elif method == 'tsne':
        extractor = TSNE(n_components=n_components)
    elif method == 'umap':
        extractor = UMAP(n_components=n_components)
    return extractor.fit_transform(X)

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

# Automated Machine Learning (AutoML)
def preprocess_data(file_path, label_name, feature_count, home_dir, 
                    impute_method='iterative', scale_method='quantile', 
                    outlier_detection=False, feature_selection=None, transform_method=None,
                    augment_data=False, handle_categorical='onehot', data_augmentation_method='smote',
                    outlier_detection_method='isolation_forest', feature_extraction_method=None,
                    categorical_encoding_method='onehot'):
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
        handle_categorical=handle_categorical,
        data_augmentation_method=data_augmentation_method,
        outlier_detection_method=outlier_detection_method,
        feature_extraction_method=feature_extraction_method,
        categorical_encoding_method=categorical_encoding_method
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

    # Apply data cleaning
    X = df_preprocessed.iloc[:, 2:].values
    X = data_cleaning(X)
    df_preprocessed.iloc[:, 2:] = X

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
                scale_method='quantile_uniform',
                outlier_detection=False,
                feature_selection='automl',
                transform_method=None,
                augment_data=True,
                handle_categorical='onehot',
                data_augmentation_method='smote',
                outlier_detection_method='isolation_forest',
                feature_extraction_method=None,
                categorical_encoding_method='onehot'
            )
