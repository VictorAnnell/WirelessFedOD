import collections
import logging
from collections.abc import Callable

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from matplotlib import pyplot as plt
import keras_cv

from dataset import load_zod2

np.random.seed(0)

class WirelessFedODSimulator:

    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.federated_train_data = None
        self.model_fn = None
        self.agent_selection_fn = self.default_agent_selection
        self.preprocess_fn = self.default_preprocess
        self.num_clients = 2
        self.num_epochs = 5
        self.batch_size = 20
        self.shuffle_buffer = 100
        self.prefetch_buffer = 10
        self.round_num = None
        self.metrics = None

        self.training_process = None
        self.train_state = None
        self.train_result = None
        self.agent_selection_fn = None
        self.sample_clients = None
        self.client_optimizer_fn = lambda: tf.keras.optimizers.SGD(learning_rate=0.02)
        self.server_optimizer_fn = lambda: tf.keras.optimizers.SGD(learning_rate=1.0)

    def set_dataset(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def set_model_fn(self, model_fn):
        self.model_fn = model_fn

    def set_hyperparameters(self,
                            num_clients=10,
                            num_epochs=5,
                            batch_size=20,
                            shuffle_buffer=100,
                            prefetch_buffer=10):
        self.num_clients = num_clients
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.prefetch_buffer = prefetch_buffer

    def set_preprocess_fn(self, preprocess_fn):
        self.preprocess_fn = preprocess_fn

    def default_preprocess(self, dataset):

        def batch_format_fn(element):
            """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
            return collections.OrderedDict(x=tf.reshape(element['pixels'],
                                                        [-1, 784]),
                                           y=tf.reshape(element['label'], [-1, 1]))

        return dataset.repeat(self.num_epochs).shuffle(self.shuffle_buffer, seed=1).batch(
            self.batch_size).map(batch_format_fn).prefetch(self.prefetch_buffer)

    def set_agent_selection_fn(self, agent_selection_fn):
        self.agent_selection_fn = agent_selection_fn

    def default_agent_selection(self, client_data, num_clients):
        return np.random.choice(client_data.client_ids, num_clients, replace=False)

    def make_federated_data(self, client_data, client_ids):
        return [
            self.preprocess_fn(client_data.create_tf_dataset_for_client(x))
            for x in client_ids
        ]

    def run_epoch(self):
        if self.train_state is None or self.round_num is None:
            self.initialize()
        self.round_num += 1
        self.sample_clients = self.agent_selection_fn(self.train_data, self.num_clients)
        self.federated_train_data = self.make_federated_data(self.train_data, self.sample_clients)
        self.train_result = self.training_process.next(self.train_state, self.federated_train_data)
        self.train_state = self.train_result.state
        self.metrics = self.train_result.metrics['client_work']['train']
        # Format metrics to be one line
        print('round {:2d}, '.format(self.round_num), end='')
        print(f'num_clients: {len(self.sample_clients)}', end=', ')
        self.print_metrics(self.train_result)

    def set_optimizers(self, client_optimizer_fn, server_optimizer_fn):
        self.client_optimizer_fn = client_optimizer_fn
        self.server_optimizer_fn = server_optimizer_fn

    def initialize(self):
        self.round_num = 0
        self.training_process = tff.learning.algorithms.build_weighted_fed_avg(
            self.model_fn,
            client_optimizer_fn=self.client_optimizer_fn,
            server_optimizer_fn=self.server_optimizer_fn)

        self.train_state = self.training_process.initialize()

    def print_metrics(self, train_result):
        for name, value in train_result.metrics['client_work']['train'].items():
            print(f'{name}: {value:.4f}', end=', ')
        print()

simulator = WirelessFedODSimulator()

# Set preprocess_fn
def preprocess_fn(dataset):
    def load_image(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        return image

    def format_element_fn(image_path, classes, bboxes):
        # Create a dictionary with the image and bounding boxes as required by KerasCV
        image = load_image(image_path)
        bounding_boxes = {
            "classes": tf.cast(classes, dtype=tf.float32),
            "boxes": bboxes,
        }
        return {"images": tf.cast(image, tf.float32), "bounding_boxes": bounding_boxes}

    def dict_to_tuple_fn(element):
        return element['images'], element['bounding_boxes']

    augmenter = tf.keras.Sequential(
        layers=[
            keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format="xyxy"),
            keras_cv.layers.RandomShear(
                x_factor=0.2, y_factor=0.2, bounding_box_format="xyxy"
            ),
            keras_cv.layers.JitteredResize(
                target_size=(640, 640), scale_factor=(0.75, 1.3), bounding_box_format="xyxy"
            ),
        ]
    )
    dataset = dataset.map(format_element_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.ragged_batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(dict_to_tuple_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # dataset = dataset.shuffle(self.shuffle_buffer)

    return dataset


simulator.set_preprocess_fn(preprocess_fn)

# Load ZOD dataset
zod_train, zod_test = load_zod2(num_clients=simulator.num_clients)
example_dataset = zod_train.create_tf_dataset_for_client(zod_train.client_ids[0])
preprocessed_example_dataset = simulator.preprocess_fn(example_dataset)

# Create model_fn
def create_kerascv_model():
    return keras_cv.models.YOLOV8Detector.from_preset("yolo_v8_m_pascalvoc", bounding_box_format="xyxy")

keras_model = create_kerascv_model()
def model_fn():
    keras_model = create_kerascv_model()
    return tff.learning.models.from_keras_model(
        keras_model,
        input_spec=preprocessed_example_dataset.element_spec,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanSquaredError()]
    )
# Set dataset, model_fn, and agent_selection_fn
simulator.set_dataset(zod_train, zod_test)
simulator.set_model_fn(model_fn)
simulator.set_agent_selection_fn(lambda x, y: np.random.choice(x.client_ids, y, replace=False))
# Set optimizers
simulator.set_optimizers(lambda: tf.keras.optimizers.Adam(learning_rate=0.001),
                         lambda: tf.keras.optimizers.Adam(learning_rate=0.1))
# Run training
simulator.run_epoch()


# if __name__ == '__main__':
def main():
    # Example Usage
    # Create simulator
    simulator = WirelessFedODSimulator()
    # Create train/test dataset
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(cache_dir='./cache')
    example_dataset = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[0])
    preprocessed_example_dataset = simulator.preprocess_fn(example_dataset)

    # Create model_fn
    def create_keras_model():
        return tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(784, )),
            tf.keras.layers.Dense(10, kernel_initializer='zeros'),
            tf.keras.layers.Softmax(),
        ])
    def model_fn():
        # We _must_ create a new model here, and _not_ capture it from an external
        # scope. TFF will call this within different graph contexts.
        keras_model = create_keras_model()
        return tff.learning.models.from_keras_model(
            keras_model,
            input_spec=preprocessed_example_dataset.element_spec,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    # Set dataset, model_fn, and agent_selection_fn
    simulator.set_dataset(emnist_train, emnist_test)
    simulator.set_model_fn(model_fn)
    simulator.set_agent_selection_fn(lambda x, y: np.random.choice(x.client_ids, y, replace=False))
    # Run training
    # while simulator.metrics is None or simulator.metrics['loss'] > 1:
    #     simulator.run_epoch()
    simulator.run_epoch()
    # Plot metrics
    # plt.plot(simulator.metrics['loss'])
    # plt.plot(simulator.metrics['sparse_categorical_accuracy'])
    # plt.show()
