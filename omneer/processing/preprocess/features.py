import os
import pandas as pd
from sklearn.feature_selection import f_classif

class Data:
    def __init__(self, label, features, csv_dir):
        self.label = label
        self.features = features
        self.csv_dir = csv_dir

    def calculate_feature_importance(self):
        df = pd.read_csv(self.csv_dir, encoding='latin1')

        # Separate the features (X) from the target (y)
        X = df[self.features].astype(bool)  # Convert the features to boolean values
        y = df[self.label]

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

    def select_top_features(self, n):
        # Call the calculate_feature_importance method
        importance_df = self.calculate_feature_importance()

        # Get the top n features
        top_n_features = importance_df.head(n)

        # Select only the top n features from the original DataFrame
        df = pd.read_csv(self.csv_dir, encoding='latin1')
        selected_features = df[top_n_features['Feature']]

        return selected_features

    def save_features_data(self, file_name, n):
        # Prepare a DataFrame
        df = self.select_top_features(n)
        df.insert(0, self.label, pd.read_csv(self.csv_dir, encoding='latin1')[self.label])

        # Extract the file name without the extension
        file_name_without_extension = os.path.splitext(file_name)[0]

        # Construct the new file name
        processed_file_name = f"{file_name_without_extension}_features_{n}.csv"

        # Set the directory path for saving the preprocessed file
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data/features')

        # Create the directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)

        # Save the DataFrame to a csv file
        output_path = os.path.join(output_dir, processed_file_name)
        df.to_csv(output_path, index=False)

if __name__ == "__main__":
    csv_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data/raw')
    for file_name in os.listdir(csv_dir):
        if file_name.endswith('.csv'):
            csv_file = os.path.join(csv_dir, file_name)

            # The name of the column in your csv file that contains the labels
            label = 'PD'

            # The names of the columns in your csv file that contain the features
            features = pd.read_csv(csv_file, encoding='latin1').columns[1:].tolist()

            # Initialize the Data object
            data = Data(
                label=label,
                features=features,
                csv_dir=csv_file,
            )

            # Save the features data to a new csv file
            data.save_features_data(file_name, 5)
