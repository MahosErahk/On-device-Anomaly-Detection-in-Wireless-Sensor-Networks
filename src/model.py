"""Baseline and Transformer multi-task autoencoders."""

import tensorflow as tf
from tensorflow.keras import Model, layers


@tf.keras.utils.register_keras_serializable(package="uv")
class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model, self.num_heads, self.d_ff, self.dropout_rate = d_model, num_heads, d_ff, dropout_rate
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([layers.Dense(d_ff, activation="relu"), layers.Dense(d_model)])
        self.norm1, self.norm2 = layers.LayerNormalization(epsilon=1e-6), layers.LayerNormalization(epsilon=1e-6)
        self.drop1, self.drop2 = layers.Dropout(dropout_rate), layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        x = self.norm1(x + self.drop1(self.attn(x, x), training=training))
        return self.norm2(x + self.drop2(self.ffn(x), training=training))

    def get_config(self):
        return {**super().get_config(), "d_model": self.d_model, "num_heads": self.num_heads,
                "d_ff": self.d_ff, "dropout_rate": self.dropout_rate}


def _heads(latent):
    def branch(units, name, activation=None):
        x = layers.Dense(units, activation="relu")(latent)
        x = layers.Dropout(0.2)(x)
        return layers.Dense(1, activation=activation, name=name)(x)
    return branch(128, "delay_output"), branch(64, "lgr_output"), branch(64, "class_output", "sigmoid")


def build_baseline_model(seq_len=1000, features=1):
    """Build the baseline CAE architecture from the Colab notebook."""
    inputs = layers.Input((seq_len, features))
    x = layers.Conv1D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling1D(2, padding="same")(x)
    x = layers.Conv1D(16, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling1D(2, padding="same")(x)
    bottleneck = layers.Conv1D(8, 3, activation="relu", padding="same", name="bottleneck")(x)
    sense = layers.Dense(128, activation="relu", name="shared_sense")(layers.Flatten()(bottleneck))
    sense = layers.Dense(64, activation="relu")(sense)
    lgr = layers.Dense(1, activation="tanh", name="lgr_output")(layers.Dense(64, activation="relu")(sense))
    delay = layers.Dense(1, name="delay_output")(layers.Dense(64, activation="relu")(sense))
    cls = layers.Dense(1, activation="sigmoid", name="class_output")(layers.Dense(64, activation="relu")(sense))
    x = layers.Conv1D(8, 3, activation="relu", padding="same")(bottleneck)
    x = layers.UpSampling1D(2)(x); x = layers.Conv1D(16, 3, activation="relu", padding="same")(x)
    x = layers.UpSampling1D(2)(x); x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
    reconstruction = layers.Conv1D(features, 3, padding="same", name="reconstruction")(x)
    return Model(inputs, [lgr, delay, cls, reconstruction], name="baseline_cae")


def build_transformer_model(seq_len=1000, features=1, latent_dim=64, num_heads=4):
    inputs = layers.Input((seq_len, features))
    x = layers.Conv1D(16, 5, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling1D(2)(x); x = layers.Conv1D(32, 5, activation="relu", padding="same")(x)
    x = layers.MaxPooling1D(2)(x); encoded = layers.Dense(latent_dim)(x)
    encoded = TransformerBlock(latent_dim, num_heads, 2 * latent_dim)(encoded)
    latent = layers.Concatenate()([layers.GlobalAveragePooling1D()(encoded), layers.GlobalMaxPooling1D()(encoded)])
    delay, lgr, cls = _heads(latent)
    x = layers.UpSampling1D(2)(encoded); x = layers.Conv1D(32, 5, activation="relu", padding="same")(x)
    x = layers.UpSampling1D(2)(x); x = layers.Conv1D(16, 5, activation="relu", padding="same")(x)
    reconstruction = layers.Conv1D(features, 1, name="reconstruction")(x)
    return Model(inputs, [delay, lgr, cls, reconstruction], name="transformer_cae")


# Exact descriptive name used in the Colab cells; retained for drop-in migration.
build_light_cir_autoencoder_with_recon = build_transformer_model


def compile_model(model, learning_rate=1e-3):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss={"delay_output": tf.keras.losses.Huber(2), "lgr_output": tf.keras.losses.Huber(.25),
              "class_output": "binary_crossentropy", "reconstruction": "mse"},
        loss_weights={"delay_output": 3., "lgr_output": 5., "class_output": 2., "reconstruction": .05})
    return model
