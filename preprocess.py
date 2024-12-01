"""
"""

import tensorflow as tf
import pandas as pd
from sklearrn.impute import KNNImputer
from typing import tuple

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
  imputer_20n = KNNImputer(n_neighbors=20)
  imputer_50n = KNNImputer(n_neighbors=50)

  knn_imputed = (
  imputer_5n.fit_transform(df)
  + imputer_10n.fit_transform(df)
  + imputer_20n.fit_transform(df)
  + imputer_50n.fit_transform(df)
  ) / 4

  knn_imputed_df = pd.DataFrame(
  knn_imputed, columns=imputer_20n.get_feature_names_out()
  )
  return knn_imputed_df


def _initial_category(sii: float) -> int:
  return 0 if sii == 0 else 1

def dataframe_to_dataset(df: pd.DataFrame, label_column: str, batch_size: int) -> tf.data.Dataset:
  df = df.copy()
  labels = df.pop(label_column)
  ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
  ds = ds.shuffle(buffer_size=len(df)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
  return ds


def preprocess_data(df:pd.DataFrame, batch_size: int) -> tuple[tf.data.Dataset, tf.data.Dataset]:
  label = 'sii_initial_category'

  df = df.loc[:, _COLUMNS_TO_TRAIN]
  df = df.rename(columns={"Basic_Demos-Age": "age",
                          "Physical-BMI": "bmi", 
                          "Physical-Height": "height", 
                          "Physical-Weight": "weight",
                          "FGC-FGC_CU":"fgc",
                          "SDS-SDS_Total_T":"sds",
                          "PreInt_EduHx-computerinternet_hoursday":"internet"
                          })
  # We keep only the rows that contain an 'sii' value.
  df = df[df["sii"].isna() == False]
  df[label] = df["sii"].apply(_initial_category)

  # We generate the validation dataset from examples 
  # that are complete from the initial training set.
  val_df = df.copy()
  val_df = val_df.dropna()
  val_df = val_df.sample(frac=0.1, random_state=42)

  train_df = df.drop(val_df.index)

  df = df.drop(columns=['sii'])
  train_df = _input_missing_data(train_df)
  
  print(f'val_df shape: {val_df.shape}')
  print(f'train_df shape: {train_df.shape}')

  # Returns the train and validation datasets.
  train_ds, val_ds = dataframe_to_dataset(train_df, label, batch_size), dataframe_to_dataset(val_df, label, batch_size)

  return (train_ds, val_ds)
