import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import numpy as np
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

def load_and_preprocess_data(file_name):
    # Load the data
    df = pd.read_csv(file_name, header=0)

    # Rename the first column as 'PD'
    df.rename(columns={0: 'PD'}, inplace=True)

    # Standardize the data (optional)
    scaler = StandardScaler()
    df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])
    
    return df

def transform_data(df):
    # Transform the data from wide to long form suitable for boxplot
    df_melt = pd.melt(df, id_vars='PD', var_name='Metabolite', value_name='Concentration')
    
    return df_melt

def create_boxplot(df_melt):
    # Create a boxplot
    plt.figure(figsize=(20, 10))
    sns.boxplot(x='Metabolite', y='Concentration', hue='PD', data=df_melt)
    plt.xticks(rotation=90)
    plt.title('Distribution of Metabolites Concentration')
    plt.show()

def calculate_correlations(df):
    # Calculate correlations
    corr = df.iloc[:, 1:].corr()

    return corr

def create_heatmap(corr):
    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation between Metabolites')
    plt.show()

def determine_grid(df, num_cols=2):
    # Determine the number of rows and columns for the subplots
    num_metabolites = len(df.columns[1:])
    num_rows = num_metabolites // num_cols
    num_rows += num_metabolites % num_cols

    return num_rows, num_cols

def create_histograms(df, num_rows, num_cols):
    # Create a figure and axes for the subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))

    # Flatten the 2D array to 1D for easier iteration
    axes = axes.ravel()

    for index, column in enumerate(df.columns[1:]):
        sns.histplot(df, x=column, hue='PD', element='step', kde=True, ax=axes[index])
        axes[index].set_title(f"Distribution of Metabolite {column}")

    # Remove any leftover subplots
    if len(df.columns[1:]) % num_cols:
        axes[-1].remove()

    plt.tight_layout()
    plt.show()

def detect_outliers(df):
    # Use the IQR method to detect and report outliers for each column
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    # Define a condition for a data point to be an outlier 
    outlier_condition = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))

    outlier_count = outlier_condition.sum(axis=0)
    print("Number of outliers in each column:\n", outlier_count)

def create_kdeplot(df):
    # Creates a KDE plot for each variable
    for column in df.columns[1:]:
        plt.figure(figsize=(10, 5))
        sns.kdeplot(data=df, x=column, hue='PD', fill=True)
        plt.title(f'KDE Plot for {column}')
        plt.show()

def create_violinplot(df_melt):
    # Create a violin plot
    plt.figure(figsize=(20, 10))
    sns.violinplot(x='Metabolite', y='Concentration', hue='PD', data=df_melt, split=True)
    plt.xticks(rotation=90)
    plt.title('Distribution of Metabolites Concentration')
    plt.show()

def create_interactive_scatters(df):
    for col in df.columns[1:]:
        fig = px.scatter(df, x=col, y="PD", color="PD", size=abs(df[col]), hover_data=df.columns)
        fig.update_layout(title=f'Interactive Scatter Plot: {col} vs PD')
        fig.show()

def create_3d_scatters(df):
    num_metabolites = len(df.columns) - 1
    fig = plt.figure(figsize=(20, num_metabolites*5))

    for i, col in enumerate(df.columns[1:], 1):
        ax = fig.add_subplot(num_metabolites, 1, i, projection='3d')
        ax.scatter3D(df[col], df["PD"], np.zeros(df[col].shape), c=df["PD"])
        ax.set_xlabel(col)
        ax.set_ylabel('PD')
        ax.set_title(f'3D Scatter Plot: {col} vs PD')

    plt.tight_layout()
    plt.show()

def create_tsne(df):
    X = df.iloc[:, 1:].values
    
    # Create a t-SNE object with custom perplexity and learning_rate
    tsne = TSNE(n_components=2, random_state=0, perplexity=30, learning_rate=200)
    
    # Fit and transform the data
    X_2d = tsne.fit_transform(X)
    
    # Apply a clustering algorithm on the reduced data
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X_2d)
    
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
    plt.show()

def create_dendrogram(df):
    linked = linkage(df.iloc[:, 1:], method='ward')

    plt.figure(figsize=(10, 7))
    dendrogram(linked, orientation='top', labels=list(df.index), distance_sort='descending', show_leaf_counts=True)
    plt.show()


def main():
    # Specify the name of the csv file
    file_name = 'Final.csv'
    
    # Load and preprocess data
    df = load_and_preprocess_data(file_name)
    
    # Transform data
    df_melt = transform_data(df)
    
    # Create boxplot
    create_boxplot(df_melt)
    
    # Calculate correlations
    corr = calculate_correlations(df)
    
    # Create heatmap
    create_heatmap(corr)
    
    # Determine the grid for subplots
    num_rows, num_cols = determine_grid(df)
    
    # Create histograms
    create_histograms(df, num_rows, num_cols)

    #Create violin plot
    create_violinplot(df_melt)

    create_tsne(df)

# Run the main function
if __name__ == "__main__":
    main()