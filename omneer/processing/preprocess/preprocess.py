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


class Data(torch.utils.data.Dataset):

    def __init__(self, label, features, csv_dir):

        self.features = features
        self.label = label
        content = self.read_csv(csv_dir)
        self.content = self.filter_incomplete_cases(content)
        self.x = [[row[k] for k in self.features] for row in self.content]
        self.y = [row[self.label] for row in self.content]
        self.x = np.array(self.x, dtype = np.float32)
        self.y = np.array(self.y, dtype = np.float32)
        self.x = self.impute_missing_values(self.x)
        #self.x = self.polynomial_features(self.x)
        #self.x = self.lasso_feature_selection(self.x, self.y)
        self.x = self.normalize_features(self.x)
        #self.x = self.pls_da_transform(self.x)

        self.save_preprocessed_data('Preprocessed_Final.csv')


    def read_csv(self, csv_file):

        content = []
        with open(csv_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                content.append(row)
        return content

    
    def filter_incomplete_cases(self, content):
        
        filtered_content = []
        for row in content:
            complete = True
            for key in self.features:
                if row[key] == '':
                    complete = False
            if complete and row[self.label] != '':
                filtered_content.append(row)
        return filtered_content


    def __len__(self):

        return len(self.content)


    def __getitem__(self, idx):

        return self.x[idx], self.y[idx]


    def input_length(self):

        return len(self.__getitem__(0)[0])
    
    @property
    def all(self):
        
        return self.x, self.y
    
    def impute_missing_values(self, x):
        imputer = IterativeImputer(max_iter=10, random_state=0)
        return imputer.fit_transform(x)

    def normalize_features(self, x):
        qt = QuantileTransformer().fit(x)
        return qt.transform(x)
    
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
        df.columns = self.features
        df[self.label] = self.y

        # Set the directory path for saving the preprocessed file
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data/preprocessing')

        # Create the directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)

        # Save the DataFrame to a csv file
        output_path = os.path.join(output_dir, file_name)
        df.to_csv(output_path, index=False)


if __name__ == "__main__":
    csv_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'data/raw')
    for file_name in os.listdir(csv_dir):
        if file_name.endswith('.csv'):
            csv_file = os.path.join(csv_dir, file_name)

            df = pd.read_csv(csv_file, encoding='latin1')

            # The name of the column in your csv file that contains the labels
            label = 'PD'

            # The names of the columns in your csv file that contain the features
            features = df.columns[1:].tolist()

            # Initialize the Data object
            data = Data(
                label=label,
                features=features,
                csv_dir=csv_file,
            )