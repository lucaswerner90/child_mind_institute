import os
import pandas as pd
from preprocess import preprocess_data

if __name__ == "__main__":
    train_df = pd.read_csv(os.path.join(os.getcwd(), "data", "train.csv"))
    preprocess_data(train_df)
