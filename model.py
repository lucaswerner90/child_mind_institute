import tensorflow as tf
import keras
from keras.utils import FeatureSpace
from tfx.v1 import 

def generate_model(ds: tf.data.Dataset) -> tf.keras.Model:
    feature_space = FeatureSpace(
        features={
            "age": FeatureSpace.float_normalized(),
            "bmi": FeatureSpace.float_normalized(),
            "fgc": FeatureSpace.float_normalized(),
            "height": FeatureSpace.float_normalized(),
            "internet": FeatureSpace.float_normalized(),
            "sds": FeatureSpace.float_normalized(),
            "weight": FeatureSpace.float_normalized(),
        },
        output_mode="concat",
    )

    train_ds_with_no_labels = ds.map(lambda x, _: x)
    feature_space.adapt(train_ds_with_no_labels)

    dict_inputs = feature_space.get_inputs()
    encoded_features = feature_space.get_encoded_features()

    x = tf.keras.layers.Dense(128, activation="relu")(encoded_features)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(dict_inputs, output)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    metrics = [
        tf.keras.metrics.Recall(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.BinaryAccuracy(),
    ]
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=metrics,
    )

    return model


def train_model(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int,
) -> tuple[keras.Model, keras.callbacks.History]:
    NUM_TRAINING_EXAMPLES = 2500
    BATCH_SIZE = 128
    STOP_POINT = 500
    TOTAL_TRAINING_EXAMPLES = int(STOP_POINT * NUM_TRAINING_EXAMPLES)
    NUM_CHECKPOINTS = 10
    STEPS_PER_EPOCH = TOTAL_TRAINING_EXAMPLES // (BATCH_SIZE * NUM_CHECKPOINTS)

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.1, patience=20, min_lr=1e-8
        ),
    ]

    history = model.fit(
        train_ds.repeat(),
        epochs=NUM_CHECKPOINTS,
        validation_data=val_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        verbose=2,
        callbacks=callbacks,
    )
    return model, history
