import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.decomposition import PCA

def load_and_clean_data(file_path):
    # Read the data
    df = pd.read_csv(file_path)

    # Handle missing data
    imputer = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Standardize data
    scaler = StandardScaler()
    df_standardized = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df.columns)
    
    return df_standardized

def perform_eda(df):
    # Summary statistics
    print(df.describe())

    # Check for missing values
    print(df.isnull().sum())

    # Count of PD diagnoses
    print(df['PD'].value_counts())

    # Histograms of metabolites
    df.drop(['Patient', 'PD'], axis=1).hist(bins=30, figsize=(15, 10))
    plt.tight_layout()
    plt.show()

    # Correlation matrix and heatmap
    corr_matrix = df.drop(['Patient'], axis=1).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.show()

    # Box plots for each metabolite by PD diagnosis
    for column in df.drop(['Patient', 'PD'], axis=1).columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x='PD', y=column, data=df)
        plt.show()

    # Pair plot of the first few features
    sns.pairplot(df, vars=df.columns[:5])
    plt.show()

    # Outlier detection and removal
    z_scores = np.abs(stats.zscore(df))
    df_no_outliers = df[(z_scores < 3).all(axis=1)]

    # Principal Component Analysis for data visualization
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df_no_outliers.drop(['Patient', 'PD'], axis=1))
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df = pd.concat([pca_df, df_no_outliers[['PD']]], axis=1)
    sns.scatterplot(x='PC1', y='PC2', hue='PD', data=pca_df)
    plt.show()
