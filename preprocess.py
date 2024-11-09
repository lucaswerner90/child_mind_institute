"""
"""

import tensorflow as tf
import pandas as pd
from sklearn.impute import KNNImputer

# These columns come from EDA step of the project. 
# For now we just decide to use the ones that contribute the most to the 'sii' values.
_COLUMNS_TO_TRAIN = [
  "Basic_Demos-Age",
  "Physical-BMI",
  "Physical-Height",
  "Physical-Weight",
  "FGC-FGC_CU",
  "SDS-SDS_Total_T",
  "PreInt_EduHx-computerinternet_hoursday",
  "sii",
  ]

def _input_missing_data(df: pd.DataFrame) -> pd.DataFrame:
  df = df.copy()
  imputer_5n = KNNImputer(n_neighbors=5)
  imputer_10n = KNNImputer(n_neighbors=10)
  imputer_50n = KNNImputer(n_neighbors=50)
  imputer_100n = KNNImputer(n_neighbors=100)
  knn_imputed = (
  imputer_5n.fit_transform(df)
  + imputer_10n.fit_transform(df)
  + imputer_50n.fit_transform(df)
  + imputer_100n.fit_transform(df)
  ) / 4
  knn_imputed_df = pd.DataFrame(
  knn_imputed, columns=imputer_50n.get_feature_names_out()
  )
  return knn_imputed_df


def _initial_category(sii: float) -> int:
  return 0 if sii == 0 else 1


def dataframe_to_dataset(df: pd.DataFrame, label_column: str) -> tf.data.Dataset:
  df = df.copy()
  labels = df.pop(label_column)
  ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
  ds = ds.shuffle(buffer_size=len(df))
  return ds


def preprocess_data(csv_file: str, batch_size: int) -> tuple(tf.data.Dataset, tf.data.Dataset):
  df = pd.read_csv(csv_file)
  df = df.loc[:, _COLUMNS_TO_TRAIN]
  df = df[not df["sii"].isna()]
  df["sii_initial_category"] = df["sii"].apply(_initial_category)

  val_df = df.sample(frac=0.2, random_state=42)
  train_df = df.drop(val_df.index)

  train_df = _input_missing_data(train_df)
  val_df = _input_missing_data(val_df)

  # Returns the train and validation datasets.
  train_ds = dataframe_to_dataset(train_df, "sii_initial_category")
  train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

  val_ds = dataframe_to_dataset(val_df, "sii_initial_category")
  val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

  return (train_ds, val_ds)
