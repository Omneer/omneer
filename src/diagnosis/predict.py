# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import pandas as pd
import subprocess

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")

    def predict(
        self,
        csv_file: Path = Input(description="Input CSV file"),
    ) -> Path:
        """Run a single prediction on the model"""
        
        # Load CSV
        df = pd.read_csv(csv_file)

        # Save cleaned df to tmp file
        tmp_file = "/tmp/temp.csv"
        df.to_csv(tmp_file, index=False)

        # Run main.py on the csv
        command = f"python main.py --csv-file {tmp_file}"
        process = subprocess.Popen(command, shell=True)
        process.wait()

        # Return results (this assumes that main.py saves results to results.csv)
        return Path("results.csv")