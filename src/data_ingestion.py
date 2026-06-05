import pandas as pd

class DataIngestion:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_data(self):
        return pd.read_csv(self.filepath)

if __name__ == "__main__":
    ingestion = DataIngestion("data/raw/dataset.csv")
    df = ingestion.load_data()
    print(df.head())