import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.feature_selection import f_classif


def load_and_preprocess_data(file_name):
    # Load the data
    df = pd.read_csv(file_name, header=0)

    # Rename the first column as 'PD'
    df.rename(columns={df.columns[1]: 'PD'}, inplace=True)

    df.iloc[:, 2:] = df.iloc[:, 2:].fillna(0)

    # Standardize the data (optional)
    scaler = StandardScaler()
    df.iloc[:, 2:] = scaler.fit_transform(df.iloc[:, 2:])
    
    return df

def transform_data(df):
    # Transform the data from wide to long form suitable for boxplot
    df_melt = pd.melt(df, id_vars='PD', var_name='Metabolite', value_name='Concentration')
    
    return df_melt

def calculate_feature_importance(X, y):
    # Calculate the F-value and p-value for each feature using ANOVA
    f_values, p_values = f_classif(X, y)
    
    # Create a DataFrame of the results
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'F-value': f_values,
        'p-value': p_values
    })

    # Sort the DataFrame by the F-value in descending order
    importance_df.sort_values('F-value', ascending=False, inplace=True)
    
    return importance_df

def select_top_features(label, features, csv_file, n):
    # Load the DataFrame from the CSV file
    df = pd.read_csv(csv_file, encoding="latin1")

    # Separate the features (X) from the target (y)
    X = df[features]
    y = df[label]

    # Call the calculate_feature_importance function
    importance_df = calculate_feature_importance(X, y)

    # Get the top n features
    top_n_features = importance_df.head(n)

    # Select only the top n features from the original DataFrame
    selected_features = df[top_n_features["Feature"]]

    return selected_features

def save_features_data(label, features, csv_file, file_name, n):
    # Prepare a DataFrame
    df = select_top_features(label, features, csv_file, n)
    df.insert(0, label, pd.read_csv(csv_file, encoding="latin1")[label])

    # Extract the file name without the extension
    file_name_without_extension = Path(file_name).stem

    # Construct the new file name
    processed_file_name = f"{file_name_without_extension}_features_{n}.csv"

    # Set the directory path for saving the preprocessed file
    output_dir = Path.cwd().parent.parent / "data" / "features"

    # Create the directory if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the DataFrame to a csv file
    output_path = output_dir / processed_file_name
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    csv_dir = Path.cwd().parent.parent / "data" / "raw"
    for csv_file in csv_dir.glob("*.csv"):
        label = "PD"
        data = load_and_preprocess_data(csv_file)
        features = data.columns[2:].tolist()

        # Save the features data to a new csv file
        save_features_data(label, features, csv_file, csv_file.name, 8)
