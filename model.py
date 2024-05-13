import numpy as np
from keras.layers import (
    GRU,
    Dense,
    Input,
)
from keras.models import Model

from preprocess import load_prepared_data

BATCH_SIZE = 32
NUM_EPOCHS = 20


def run_model():
    scalar_sequences, categorical_sequences, labels = load_prepared_data()
    input_sequences = np.concatenate([categorical_sequences, scalar_sequences], axis=2)

    print("loaded data")
    sequences = Input(shape=input_sequences.shape[1:])
    lstm_output = GRU(128, return_sequences=True)(sequences)
    x = Dense(128, activation="relu")(lstm_output)
    x = Dense(32, activation="relu")(x)

    # Output layer
    output = Dense(1, activation="sigmoid")(x)

    # Model
    model = Model(inputs=sequences, outputs=output)
    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["mean_absolute_error"],
    )

    # model.summary()

    # Shuffling data is important because the data is ordered by student_id, so earlier
    # data is of students who joined the platform earlier and therefore would have a
    # different distribution of problems solved
    indices = np.arange(len(input_sequences))
    np.random.shuffle(indices)
    np.save("models/indices.npy", indices)
    input_sequences = input_sequences[indices]
    labels = labels[indices]

    train_split_percent = 0.8
    train_sequences, train_labels = (
        input_sequences[: int(train_split_percent * len(input_sequences))],
        labels[: int(train_split_percent * len(labels))],
    )
    test_sequences, test_labels = (
        input_sequences[int(train_split_percent * len(input_sequences)) :],
        labels[int(train_split_percent * len(labels)) :],
    )

    model.fit(
        train_sequences,
        train_labels,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        validation_split=0.2,
    )
    model.evaluate(test_sequences, test_labels)
    model.save("models/predictor.keras")


if __name__ == "__main__":
    run_model()
