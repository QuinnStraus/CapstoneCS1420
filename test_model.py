import keras
import numpy as np

from preprocess import load_prepared_data


def naive_predictor():
    scalar_sequences, _, labels = load_prepared_data()
    # Past problem difficulty is the last column of the scalar sequences
    difficulties = scalar_sequences[:, :, -1]
    print(difficulties.mean())
    # onehot_difficulties = np.zeros((difficulties.shape[0], difficulties.shape[1], 3))
    # onehot_difficulties[difficulties < EASY_THRESHOLD, 0] = 1
    # onehot_difficulties[
    #     (EASY_THRESHOLD <= difficulties) & (difficulties < MEDIUM_THRESHOLD), 1
    # ] = 1
    # onehot_difficulties[difficulties >= MEDIUM_THRESHOLD, 2] = 1
    # print(onehot_difficulties.sum(axis=(0, 1)))
    print(
        "use prior difficulty as prediction: accuracy:",
        np.abs(difficulties - labels).mean(),
    )


def test_model():
    model: keras.Model = keras.saving.load_model(
        "models/predictor.keras", compile=False
    )
    indices = np.load("models/indices.npy")
    scalar_sequences, categorical_sequences, labels = load_prepared_data()
    input_sequences = np.concatenate([categorical_sequences, scalar_sequences], axis=2)
    input_sequences = input_sequences[indices]
    labels = labels[indices]
    test_sequences, test_labels = (
        input_sequences[int(0.8 * len(input_sequences)) :],
        labels[int(0.8 * len(labels)) :],
    )
    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["mean_absolute_error"],
    )
    model.evaluate(test_sequences, test_labels)
    out: np.ndarray = model.predict(test_sequences)
    print(
        "average",
        test_labels.mean(),
        "difference",
        np.abs(out.squeeze() - test_labels).mean(axis=0),
    )


if __name__ == "__main__":
    naive_predictor()
    test_model()
