import nest_asyncio

nest_asyncio.apply()

import argparse
import collections
import numpy as np
import random
import tensorflow_federated as tff
import tensorflow as tf

np.random.seed(0)
NUM_ROUNDS = 100
NUM_CLIENTS = 100
NUM_TOTAL_CLIENTS = 3383
NUM_EPOCHS = 5
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10


def preprocess(dataset):
    def batch_format_fn(element):
        """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
        return collections.OrderedDict(
            x=tf.reshape(element["pixels"], [-1, 784]),
            y=tf.reshape(element["label"], [-1, 1]),
        )

    return (
        dataset.repeat(NUM_EPOCHS)
        .shuffle(SHUFFLE_BUFFER, seed=1)
        .batch(BATCH_SIZE)
        .map(batch_format_fn)
        .prefetch(PREFETCH_BUFFER)
    )


def make_federated_data(client_data, client_ids):
    return [preprocess(client_data.create_tf_dataset_for_client(x)) for x in client_ids]


def create_keras_model():
    return tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(784,)),
            tf.keras.layers.Dense(10, kernel_initializer="zeros"),
            tf.keras.layers.Softmax(),
        ]
    )


def main(args):
    tff.federated_computation(lambda: "Hello, World!")()
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

    example_dataset = emnist_train.create_tf_dataset_for_client(
        emnist_train.client_ids[0]
    )

    preprocessed_example_dataset = preprocess(example_dataset)

    def model_fn():
        # We _must_ create a new model here, and _not_ capture it from an external
        # scope. TFF will call this within different graph contexts.
        keras_model = create_keras_model()
        return tff.learning.from_keras_model(
            keras_model,
            input_spec=preprocessed_example_dataset.element_spec,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )

    if args.secagg:
        print("Using Secure Aggregation")
        model_aggregator = tff.learning.secure_aggregator(zeroing=False, clipping=False)
    else:
        print("!!! NOT Using Secure Aggregation !!!")
        model_aggregator = tff.aggregators.MeanFactory()
    iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
        model_aggregator=model_aggregator,
    )

    # Begin TFF Process:
    state = iterative_process.initialize()
    for round_num in range(1, NUM_ROUNDS):
        sample_ids = random.sample(range(NUM_TOTAL_CLIENTS), NUM_CLIENTS)
        sample_clients = [emnist_train.client_ids[i] for i in sample_ids]
        federated_train_data = make_federated_data(emnist_train, sample_clients)

        result = iterative_process.next(state, federated_train_data)
        state = result.state
        metrics = result.metrics
        print(f"round {round_num:2d}, metrics={metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--secagg",
        dest="secagg",
        action="store_true",
        help="Choose to use Secure Aggregation",
    )
    parser.set_defaults(secagg=False)

    args = parser.parse_args()
    main(args)
