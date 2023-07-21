import pandas as pd
from pathlib import Path
from sklearn.model_selection import cross_val_score
from omneer_cli.processing.preprocess.data import Data
from omneer_cli.processing.preprocess.utils import data_cleaning
from omneer_cli.processing.preprocess.feature_selection import permutation_importance_feature_selection

# Automated Machine Learning (AutoML)
def preprocess_data(file_path, label_name, feature_count, home_dir, 
                    impute_method='iterative', scale_method='quantile', 
                    outlier_detection=False, feature_selection=None, transform_method=None,
                    augment_data=False, handle_categorical='onehot', data_augmentation_method='smote',
                    outlier_detection_method='isolation_forest', feature_extraction_method=None,
                    categorical_encoding_method='onehot'):
    df = pd.read_csv(file_path, encoding='latin1')

    # Extract all the columns. The label column is the second column in the data.
    all_cols = df.columns.tolist()

    # But remove the patient and label columns from the features we want to preprocess
    features = all_cols[2:feature_count+2]

    data_preprocess = Data(
        label=label_name,
        features=features,
        csv_dir=file_path,
        home_dir=home_dir,
        impute_method=impute_method,
        scale_method=scale_method,
        outlier_detection=outlier_detection,
        feature_selection=feature_selection,
        transform_method=transform_method,
        augment_data=augment_data,
        handle_categorical=handle_categorical,
        data_augmentation_method=data_augmentation_method,
        outlier_detection_method=outlier_detection_method,
        feature_extraction_method=feature_extraction_method,
        categorical_encoding_method=categorical_encoding_method
    )

    # Prepare a DataFrame
    x, y = data_preprocess.all
    df_preprocessed = pd.DataFrame(x)

    # Add the label as the first column
    df_preprocessed.insert(0, label_name, y)

    # If PCA or linear feature selection was applied, rename the transformed features
    if data_preprocess.feature_selection == 'automl':
        df_preprocessed.columns = [label_name] + [f'automl_feature_{i}' for i in range(df_preprocessed.shape[1]-1)]
    else:
        df_preprocessed.columns = [label_name] + features

    # Add back the 'Patient' column to the first position of the preprocessed dataframe
    df_preprocessed.insert(0, 'Patient', df['Patient'])

    # Apply data cleaning
    X = df_preprocessed.iloc[:, 2:].values
    X = data_cleaning(X)
    df_preprocessed.iloc[:, 2:] = X

    # Extract the stub of the file path without the extension
    file_name_without_extension = Path(file_path).stem

    # Save the DataFrame to a csv file
    output_path = home_dir / 'omneer_files' / 'data' / 'preprocessed' / f"{file_name_without_extension}_preprocessed.csv"
    df_preprocessed.to_csv(output_path, index=False)

    return output_path

if __name__ == "__main__":
    home_dir = Path.home()
    csv_dir = home_dir / 'omneer_files' / 'data' / 'raw'
    for file_path in csv_dir.iterdir():
        if file_path.name.endswith('.csv'):
            df = pd.read_csv(file_path, encoding='latin1')

            patient = 'Patient'
            label = 'PD'
            features = df.columns[2:].tolist()

            preprocess_data(
                file_path=file_path,
                label_name=label,
                feature_count=len(features),
                home_dir=home_dir,
                impute_method='iterative',
                scale_method='quantile_uniform',
                outlier_detection=False,
                feature_selection='automl',
                transform_method=None,
                augment_data=True,
                handle_categorical='onehot',
                data_augmentation_method='smote',
                outlier_detection_method='isolation_forest',
                feature_extraction_method=None,
                categorical_encoding_method='onehot'
            )
