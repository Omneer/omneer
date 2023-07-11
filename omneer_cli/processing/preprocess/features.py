
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def load_and_preprocess_data(file_name):
    """
    Load and preprocess data from a CSV file.
    
    The function performs the following preprocessing steps:
    - Renames the first column to 'PD'
    - Fills missing values with 0
    - Standardizes the features

    Parameters
    ----------
    file_name : str
        Path to the CSV file.

    Returns
    -------
    df : pandas.DataFrame
        Preprocessed data.
    """
    df = pd.read_csv(file_name, header=0)
    df.rename(columns={df.columns[1]: 'PD'}, inplace=True)
    df.iloc[:, 2:] = df.iloc[:, 2:].fillna(0)

    return df


def transform_data(df):
    """
    Transform data from wide to long format.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Original data in wide format.

    Returns
    -------
    df_melt : pandas.DataFrame
        Transformed data in long format.
    """
    df_melt = pd.melt(df, id_vars='PD', var_name='Metabolite', value_name='Concentration')
    
    return df_melt


def calculate_feature_importance(X, y):
    """
    Calculate the importance of each feature using ANOVA.
    
    Parameters
    ----------
    X : pandas.DataFrame
        Features.
    y : pandas.Series
        Target variable.

    Returns
    -------
    importance_df : pandas.DataFrame
        Dataframe containing the F-value and p-value for each feature.
    """
    f_values, p_values = f_classif(X, y)
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'F-value': f_values,
        'p-value': p_values
    })
    importance_df.sort_values('F-value', ascending=False, inplace=True)
    
    return importance_df


def select_top_features(label, features, csv_file, n):
    """
    Select the top n features based on importance.
    
    Parameters
    ----------
    label : str
        Name of the target variable.
    features : list of str
        List of features.
    csv_file : str
        Path to the CSV file.
    n : int
        Number of top features to select.

    Returns
    -------
    selected_features : pandas.DataFrame
        Dataframe containing the selected features.
    """
    df = pd.read_csv(csv_file, encoding="latin1")
    X = df[features]
    y = df[label]
    importance_df = calculate_feature_importance(X, y)
    top_n_features = importance_df.head(n)
    selected_features = df[top_n_features["Feature"]]
    
    return selected_features

def save_features_data(label, features, csv_file, file_name, n, features_dir):
    """
    Prepare a DataFrame with selected features and save it as a CSV file.

    Parameters
    ----------
    label : str
        Name of the target variable.
    features : list of str
        List of features.
    csv_file : str
        Path to the CSV file.
    file_name : str
        Name of the output file.
    n : int
        Number of top features to select.
    features_dir : Path
        Path to the features directory.
    """
    df = select_top_features(label, features, csv_file, n)
    original_df = pd.read_csv(csv_file, encoding="latin1")
    patient_column = original_df.iloc[:, 0]  # Get the original first column
    df.insert(0, "Patient", patient_column)  # Add the patient column to the DataFrame
    df.insert(1, "PD", original_df['PD'])  # Add back the 'PD' column
    file_name_without_extension = Path(file_name).stem
    processed_file_name = f"{file_name_without_extension}_features_{n}.csv"
    output_dir = features_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / processed_file_name
    df.to_csv(str(output_path), index=False)

def process_data(input_file, n, features_dir):
    """
    Load, preprocess, select top features, and save the processed data.

    Parameters
    ----------
    input_file : str
        Path to the input CSV file.
    n : int
        Number of top features to select.
    features_dir : Path
        Path to the features directory.
    """
    label = "PD"
    features = load_and_preprocess_data(input_file).columns[2:].tolist()
    save_features_data(label, features, input_file, input_file.name, n, features_dir)

if __name__ == "__main__":
    csv_dir = Path.cwd().parent.parent / "data" / "raw"
    for csv_file in csv_dir.glob("*.csv"):
        process_data(csv_file, 8)
