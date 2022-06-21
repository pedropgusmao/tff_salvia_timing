import os

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

import flwr as fl
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from time import time


np.random.seed(0)

NUM_ROUNDS = 10
NUM_CLIENTS = 100
NUM_TOTAL_CLIENTS = 3383
NUM_EPOCHS = 5
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10

# Load EMNIST
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()


# Preprocess function
def preprocess(dataset):
    def batch_format_fn(element):
        """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
        # return collections.OrderedDict(
        return (
            tf.reshape(element["pixels"], [-1, 784]),
            tf.reshape(element["label"], [-1, 1]),
        )

    return (
        dataset.repeat(NUM_EPOCHS)
        .shuffle(SHUFFLE_BUFFER, seed=1)
        .batch(BATCH_SIZE)
        .map(batch_format_fn)
        .prefetch(PREFETCH_BUFFER)
    )


class FlwrClient(fl.client.NumPyClient):
    def __init__(self, model, train_dataset, val_dataset) -> None:
        super().__init__()
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        num_samples = sum(1 for _ in self.train_dataset)
        processed_data = preprocess(self.train_dataset)
        self.model.fit(processed_data, verbose=2)
        return self.model.get_weights(), num_samples, {}

    def evaluate(self, parameters, config):
        num_samples = sum(1 for _ in self.val_dataset)
        processed_data = preprocess(self.val_dataset)
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(processed_data, verbose=2)
        return loss, num_samples, {"accuracy": acc}


def client_fn(cid: str) -> fl.client.Client:
    # Load model
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(784,)),
            tf.keras.layers.Dense(10, kernel_initializer="zeros"),
            tf.keras.layers.Softmax(),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.02),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    # Load data partition (divide MNIST into NUM_CLIENTS distinct partitions)
    train_dataset = emnist_train.create_tf_dataset_for_client(
        emnist_train.client_ids[int(cid)]
    )
    val_dataset = emnist_train.create_tf_dataset_for_client(
        emnist_test.client_ids[int(cid)]
    )
    # Create and return client
    return FlwrClient(model, train_dataset, val_dataset)


def main() -> None:

    # Start Flower simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_TOTAL_CLIENTS,
        client_resources={"num_cpus": 2},
        num_rounds=NUM_ROUNDS,
        strategy=fl.server.strategy.FedAvg(
            fraction_fit=NUM_CLIENTS / NUM_TOTAL_CLIENTS,
            fraction_eval=0.1,
            min_fit_clients=NUM_CLIENTS,
            min_eval_clients=10,
            min_available_clients=NUM_CLIENTS,
        ),
    )


if __name__ == "__main__":
    start = time()
    main()
    end = time()
    print(f"Flower without SecAgg took {end - start}.")
