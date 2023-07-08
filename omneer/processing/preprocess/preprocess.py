import torch
import csv
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import QuantileTransformer, RobustScaler, PowerTransformer
from sklearn.impute import KNNImputer
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import Lasso
import os
from tqdm import tqdm

class Data(torch.utils.data.Dataset):
    def __init__(self, label, features, csv_dir):
        self.features = features
        self.label = label
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
        x = self.impute_missing_values(x)
        x = self.normalize_features(x)
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
    
    def impute_missing_values(self, x):
        with tqdm(total=2, desc="Imputing missing values") as pbar:
            pbar.set_postfix(stage="Iterative Imputer")
            imputer = IterativeImputer(max_iter=10, random_state=0)

            # Divide the iterative imputation process into smaller steps
            num_steps = 10
            num_rows = x.shape[0]
            step_size = num_rows // num_steps

            for i in range(num_steps):
                start = i * step_size
                end = (i + 1) * step_size
                imputed_x = imputer.fit_transform(x[start:end])
                x[start:end] = imputed_x
                pbar.update(1 / num_steps)

            pbar.set_postfix(stage="KNN Imputer")
            knn_imputer = KNNImputer()
            x = knn_imputer.fit_transform(x)
            pbar.update(1)
        return x

    def normalize_features(self, x):
        with tqdm(total=1, desc="Normalizing features") as pbar:
            pbar.set_postfix(stage="Quantile Transformer")
            qt = QuantileTransformer()
            x = qt.fit_transform(x)
            pbar.update(1)
        return x
    
    def pls_da_transform(self, x):
        pls_da = PLSRegression(n_components=10)
        return pls_da.fit_transform(x, self.y)

    def polynomial_features(self, x):
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        return poly.fit_transform(x)
        
    def lasso_feature_selection(self, x, y):
        lasso = Lasso(alpha=0.1)
        lasso.fit(x, y)
        return x[:, lasso.coef_ != 0]

    def save_preprocessed_data(self, file_name):
        # Prepare a DataFrame
        df = pd.DataFrame(self.x)
        df.insert(0, self.label, self.y)
        df.columns = [self.label] + self.features

        # Extract the file name without the extension
        file_name_without_extension = os.path.splitext(file_name)[0]

        # Set the directory path for saving the preprocessed file
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data/preprocessing')

        # Create the directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)

        # Save the DataFrame to a csv file
        output_path = os.path.join(output_dir, f"{file_name_without_extension}.csv")  # Changed this line
        df.to_csv(output_path, index=False)



if __name__ == "__main__":
    csv_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data/raw')
    for file_name in os.listdir(csv_dir):
        if file_name.endswith('.csv'):
            csv_file = os.path.join(csv_dir, file_name)

            df = pd.read_csv(csv_file, encoding='latin1')

            patient = 'Patient'

            # The name of the column in your csv file that contains the labels
            label = 'PD'

            # The names of the columns in your csv file that contain the features
            features = df.columns[2:].tolist()

            # Initialize the Data object
            data = Data(
                label=label,
                features=features,
                csv_dir=csv_file,
            )

            # Save the preprocessed data to a new csv file
            data.save_preprocessed_data(file_name)
