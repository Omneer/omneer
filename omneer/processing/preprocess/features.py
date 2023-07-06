import pandas as pd
from sklearn.feature_selection import f_classif
from pathlib import Path
import warnings

def calculate_feature_importance(label, features, csv_file):
    df = pd.read_csv(csv_file, encoding="latin1")

    # Check and remove the columns with constant values
    df = df.loc[:,df.apply(pd.Series.nunique) != 1]

    # Get updated features after removing constant columns
    features = [feat for feat in features if feat in df.columns]

    # Separate the features (X) from the target (y)
    X = df[features].astype(bool)  # Convert the features to boolean values
    y = df[label]

    # Calculate the F-value and p-value for each feature using ANOVA
    with warnings.catch_warnings():  # Ignore the warnings
        warnings.filterwarnings('ignore')
        try:
            f_values, p_values = f_classif(X, y)
        except ValueError:  # Constant features produce ValueError in f_classif
            return None

    # Create a DataFrame of the results
    importance_df = pd.DataFrame({"Feature": X.columns, "F-value": f_values, "p-value": p_values})

    # Sort the DataFrame by the F-value in descending order
    importance_df.sort_values("F-value", ascending=False, inplace=True)

    return importance_df

def select_top_features(label, features, csv_file, n):
    # Call the calculate_feature_importance function
    importance_df = calculate_feature_importance(label, features, csv_file)

    # Get the top n features
    top_n_features = importance_df.head(n)

    # Select only the top n features from the original DataFrame
    df = pd.read_csv(csv_file, encoding="latin1")
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
        features = pd.read_csv(csv_file, encoding="latin1").columns[1:].tolist()

        # Save the features data to a new csv file
        save_features_data(label, features, csv_file, csv_file.name, 8)
