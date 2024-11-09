import tensorflow as tf


def encode_numerical_features(feature, name, dataset):
    # Create a Normalization layer for our numerical feature
    normalizer = tf.keras.layers.Normalization()

    # We get the feature from our dataset
    feature_ds = dataset.map(lambda x, _: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    normalizer.adapt(feature_ds)

    encoded_feature = normalizer(feature)
    return encoded_feature
