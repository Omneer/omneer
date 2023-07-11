import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from typing import Tuple, Union
import numpy as np
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from pathlib import Path
from sklearn.feature_selection import f_classif


# Specify the output directory for saving figures
output_dir = Path.cwd().parent / 'visualization/raw'


def load_and_preprocess_data(file_name: Path) -> pd.DataFrame:
    # Load the data
    df = pd.read_csv(file_name, header=0)

    # Rename the first column as 'PD'
    df.rename(columns={0: 'PD'}, inplace=True)

    df.iloc[:, 2:] = df.iloc[:, 2:].fillna(0)

    # Standardize the data (optional)
    scaler = StandardScaler()
    df.iloc[:, 2:] = scaler.fit_transform(df.iloc[:, 2:])

    return df


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    # Transform the data from wide to long form suitable for boxplot
    df_melt = pd.melt(df, id_vars='PD', var_name='Metabolite', value_name='Concentration')

    return df_melt


def create_boxplot(df_melt: pd.DataFrame):
    # Create a boxplot
    plt.figure(figsize=(20, 10))
    sns.boxplot(x='Metabolite', y='Concentration', hue='PD', data=df_melt)
    plt.xticks(rotation=90)
    plt.title('Distribution of Metabolites Concentration')
    plt.savefig(output_dir / 'boxplot.png')

def calculate_correlations(df: pd.DataFrame) -> pd.DataFrame:
    # Calculate correlations
    corr = df.iloc[:, 2:].corr()

    return corr


def create_heatmap(corr: pd.DataFrame):
    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation between Metabolites')
    plt.savefig(output_dir / 'heatmap.png')


def determine_grid(df: pd.DataFrame, num_cols: int=2) -> Tuple[int, int]:
    # Determine the number of rows and columns for the subplots
    num_metabolites = len(df.columns[2:])
    num_rows = num_metabolites // num_cols
    num_rows += num_metabolites % num_cols

    return num_rows, num_cols


def create_histograms(df: pd.DataFrame, num_rows: int, num_cols: int):
    # Create a figure and axes for the subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))

    # Flatten the 2D array to 1D for easier iteration
    axes = axes.ravel()

    for index, column in enumerate(df.columns[2:]):
        sns.histplot(df, x=column, hue='PD', element='step', kde=True, ax=axes[index])
        axes[index].set_title(f"Distribution of Metabolite {column}")

    # Remove any leftover subplots
    if len(df.columns[2:]) % num_cols:
        axes[-1].remove()

    plt.tight_layout()
    plt.savefig(output_dir / 'histograms.png')


def detect_outliers(df: pd.DataFrame):
    # Use the IQR method to detect and report outliers for each column
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    # Define a condition for a data point to be an outlier 
    outlier_condition = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))

    outlier_count = outlier_condition.sum(axis=0)
    print("Number of outliers in each column:\n", outlier_count)


def create_kdeplot(df: pd.DataFrame):
    # Creates a KDE plot for each variable
    for column in df.columns[2:]:
        plt.figure(figsize=(10, 5))
        sns.kdeplot(data=df, x=column, hue='PD', fill=True)
        plt.title(f'KDE Plot for {column}')


def create_violinplot(df_melt: pd.DataFrame):
    # Create a violin plot
    plt.figure(figsize=(20, 10))
    sns.violinplot(x='Metabolite', y='Concentration', hue='PD', data=df_melt, split=True)
    plt.xticks(rotation=90)
    plt.title('Distribution of Metabolites Concentration')
    plt.savefig(output_dir / 'violinplot.png')

def create_interactive_scatters(df: pd.DataFrame):
    for col in df.columns[2:]:
        fig = px.scatter(df, x=col, y="PD", color="PD", size=abs(df[col]), hover_data=df.columns)
        fig.update_layout(title=f'Interactive Scatter Plot: {col} vs PD')
        fig.show()


def create_3d_scatters(df: pd.DataFrame):
    num_metabolites = len(df.columns) - 1
    fig = plt.figure(figsize=(20, num_metabolites*5))

    for i, col in enumerate(df.columns[2:], 1):
        ax = fig.add_subplot(num_metabolites, 1, i, projection='3d')
        ax.scatter3D(df[col], df["PD"], np.zeros(df[col].shape), c=df["PD"])
        ax.set_xlabel(col)
        ax.set_ylabel('PD')
        ax.set_title(f'3D Scatter Plot: {col} vs PD')

    plt.tight_layout()


def create_tsne(df: pd.DataFrame):
    X = df.iloc[:, 2:].values

    # Create a t-SNE object with custom perplexity and learning_rate
    tsne = TSNE(n_components=2, random_state=0, perplexity=30, learning_rate=200)

    # Fit and transform the data
    X_2d = tsne.fit_transform(X)

    # Apply a clustering algorithm on the reduced data
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(X_2d)  # Set n_init explicitly

    # Create a new dataframe for plot
    df_tsne = pd.DataFrame(X_2d, columns=['Component 1', 'Component 2'])
    df_tsne['PD'] = df['PD']
    df_tsne['Cluster'] = kmeans.labels_

    # Create a 2D scatter plot
    plt.figure(figsize=(8, 6))
    colors = ['r', 'g']

    for i, color in zip(df_tsne['PD'].unique(), colors):
        plt.scatter(df_tsne.loc[df_tsne['PD'] == i, 'Component 1'], df_tsne.loc[df_tsne['PD'] == i, 'Component 2'],
                    c=color, label=f"PD = {i}")

    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.title('2D t-SNE')
    plt.savefig(output_dir / 'tsne.png')


def create_pairplot(df: pd.DataFrame):
    sns.pairplot(df, hue='PD')
    plt.savefig(output_dir / 'pairplot.png')


def create_dendrogram(df: pd.DataFrame):
    linked = linkage(df.iloc[:, 2:], method='ward')

    plt.figure(figsize=(10, 7))
    dendrogram(linked, orientation='top', labels=list(df.index), distance_sort='descending', show_leaf_counts=True)


def calculate_feature_importance(df: pd.DataFrame) -> pd.DataFrame:
    # Separate the features (X) from the target (y)
    X = df.iloc[:, 2:]
    y = df['PD']

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


def plot_feature_importance(importance_df: pd.DataFrame):
    # Select the top 20 features
    top_20_features = importance_df.head(20)

    # Create a bar plot of the F-values for the top 20 features
    plt.figure(figsize=(10, 10))  # Increase the plot size to accommodate more features
    sns.barplot(x='F-value', y='Feature', data=top_20_features)
    plt.title('Top 20 Feature Importance Ranking Based on Variance')  # Update the plot title
    plt.xlabel('F-value (Variance)')
    plt.ylabel('Feature')
    plt.tight_layout()

    # Define the directories
    home_dir = Path.home()  # Get the home directory
    omneer_files_dir = home_dir / "omneer_files"
    vis_dir = omneer_files_dir / "visualization"
    raw_dir = vis_dir / "raw"

    # Create directories if they don't exist
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Define the file path
    file_path = raw_dir / 'feature_importance.png'

    # Save the figure
    plt.savefig(file_path)

def main():
    # Specify the directory path where the CSV files are stored
    csv_dir = Path.cwd().parent / 'data/raw'

    # Iterate over the files in the directory
    for file_name in csv_dir.glob('*.csv'):

        # Load and preprocess data
        df = load_and_preprocess_data(file_name)

        # Transform data
        df_melt = transform_data(df)

        # Create boxplot
        #create_boxplot(df_melt)

        # Create pairplot
        #create_pairplot(df)

        # Calculate correlations
        corr = calculate_correlations(df)

        # Create heatmap
        #create_heatmap(corr)

        # Determine the grid for subplots
        num_rows, num_cols = determine_grid(df)

        # Create histograms
        #create_histograms(df, num_rows, num_cols)

        # Create violin plot
        #create_violinplot(df_melt)

        # Create t-SNE plot
        #create_tsne(df)

        # Calculate feature importance
        importance_df = calculate_feature_importance(df)
            
        # Plot feature importance
        plot_feature_importance(importance_df)


# Run the main function
if __name__ == "__main__":
    main()

