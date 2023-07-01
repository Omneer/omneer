# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 15:41:25 2022

@author: Diana
"""

import torch
import csv
import numpy as np
from torch.utils.data import DataLoader
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import QuantileTransformer, RobustScaler, PowerTransformer
from sklearn.impute import KNNImputer

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
        self.x = self.normalize_features(self.x)


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
        x_qt = qt.transform(x)
        scaler = RobustScaler().fit(x_qt)
        return scaler.transform(x_qt)

if __name__ == "__main__":
    
    import pandas as pd

    df = pd.read_csv('Final.csv', encoding='latin1')
    data = Data(
        label = 'PD',
        features = df.iloc[:, 1:-1],
        csv_dir = 'Final.csv',
    )


    print(df)
    
    """
    
    
    
    dataloader = DataLoader(data,
                             batch_size=4,
                             shuffle=True)
    
    
    for batch_input, batch_label in dataloader:
         print(batch_input.shape, batch_label.shape)
         print(batch_input)
         print(batch_label)

    """
