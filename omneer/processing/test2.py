import os
import sys
import pandas as pd
from sklearn.feature_selection import f_classif
from omneer.processing.preprocess.features import FeaturesData

def main(csv_file, num_features):
    label = 'PD'
    features = pd.read_csv(csv_file, encoding='latin1').columns[1:].tolist()

    data = FeaturesData(
        label=label,
        features=features,
        csv_dir=csv_file,
    )

    selected_features_csv = data.get_selected_features_csv(csv_file, num_features)
    print(f"New CSV file with selected features: {selected_features_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <csv_file> <num_features>")
        sys.exit(1)

    csv_file = sys.argv[1]
    num_features = int(sys.argv[2])
    main(csv_file, num_features)
