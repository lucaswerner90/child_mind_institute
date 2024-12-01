import os
import pandas as pd
import tensorflow as tf
from preprocess import preprocess_data
from model import generate_model, train_model

if __name__ == "__main__":
    BATCH_SIZE=128
    EPOCHS = 500

    train_df = pd.read_csv(os.path.join(os.getcwd(), "data", "train.csv"))
    train_ds, val_ds = preprocess_data(train_df, batch_size=BATCH_SIZE)
    model = generate_model(train_ds)
    model, history = train_model(model, train_ds, val_ds, epochs=EPOCHS)
    sample = {
        "age": 9,
        "bmi": 14,
        "fgc": 3,
        "height": 48,
        "internet": 0,
        "sds": 64,
        "weight": 46,
    }

    input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
    predictions = model.predict(input_dict)
    print(f'Predictions: {predictions}')