from keras.models import Model
from keras.optimizers import Adam
from keras.layers import (
    Input,
    LSTM,
    Dropout,
    Dense,
    Concatenate,
    GRU,
    Normalization,
)

from preprocess import load_prepared_data
import numpy as np

BATCH_SIZE = 32


def run_model():
    scalar_sequences, categorical_sequences, labels = load_prepared_data()
    difficulties = scalar_sequences[:, 1:, -1].mean(axis=1)
    onehot_difficulties = np.zeros((len(difficulties), 3))
    onehot_difficulties[difficulties < 1 / 3, 0] = 1
    onehot_difficulties[(1 / 3 <= difficulties) & (difficulties < 2 / 3), 1] = 1
    onehot_difficulties[difficulties >= 2 / 3, 2] = 1
    print(((onehot_difficulties == 1) & (labels == 1)).sum() / len(labels))
    print("loaded data")
    # Categorical input and embedding
    categorical_input = Input(
        shape=(categorical_sequences.shape[1], categorical_sequences.shape[2])
    )

    # Scalar input
    scalar_input = Input(shape=scalar_sequences.shape[1:])
    normalization = Normalization()
    normalization.adapt(scalar_sequences)
    normalized_scalar = normalization(scalar_input)
    categorical_dense = Dense(50)(categorical_input)
    combined = Concatenate()([categorical_dense, normalized_scalar])

    # Combine the two inputs
    lstm_output = GRU(64)(combined)
    # Add additional layers
    x = Dense(64, activation="relu")(lstm_output)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu")(x)

    # Output layer
    output = Dense(3, activation="softmax")(x)

    # Model
    model = Model(inputs=[categorical_input, scalar_input], outputs=output)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # model.summary()

    train_split_percent = 0.8
    train_categorical, train_scalar, train_labels = (
        categorical_sequences[: int(train_split_percent * len(categorical_sequences))],
        scalar_sequences[: int(train_split_percent * len(scalar_sequences))],
        labels[: int(train_split_percent * len(labels))],
    )
    test_categorical, test_scalar, test_labels = (
        categorical_sequences[int(train_split_percent * len(categorical_sequences)) :],
        scalar_sequences[int(train_split_percent * len(scalar_sequences)) :],
        labels[int(train_split_percent * len(labels)) :],
    )

    model.fit(
        [train_categorical, train_scalar],
        train_labels,
        batch_size=BATCH_SIZE,
        epochs=10,
        validation_split=0.2,
    )

    print(model.evaluate([test_categorical, test_scalar], test_labels))


run_model()