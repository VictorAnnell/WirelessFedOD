import collections
import datetime

import keras
import keras_cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from car import Car
from dataset import class_mapping
from tqdm.auto import tqdm

# keras.mixed_precision.set_global_policy('mixed_float16')
np.random.seed(0)

BATCH_SIZE = 4


def visualize_dataset(inputs, value_range, rows, cols, bounding_box_format):
    inputs = next(iter(inputs.take(1)))
    images, bounding_boxes = inputs
    keras_cv.visualization.plot_bounding_box_gallery(
        images,
        value_range=value_range,
        rows=rows,
        cols=cols,
        y_true=bounding_boxes,
        scale=5,
        font_scale=0.7,
        bounding_box_format=bounding_box_format,
        class_mapping=class_mapping,
    )
    plt.show()


def visualize_detections(model, dataset, bounding_box_format="xyxy"):
    # dataset = dataset.unbatch()
    # dataset.ragged_batch(4)
    dataset = dataset.shuffle(100)
    images, y_true = next(iter(dataset.take(1)))
    y_pred = model.predict(images)
    keras_cv.visualization.plot_bounding_box_gallery(
        images,
        value_range=(0, 255),
        bounding_box_format=bounding_box_format,
        y_true=y_true,
        y_pred=y_pred,
        scale=4,
        rows=2,
        cols=2,
        show=True,
        font_scale=0.7,
        class_mapping=class_mapping,
        legend=True,
    )
    plt.show()

def weight_scalling_factor(cars, car):
    # client_names = list(cars.keys())
    #get the bs
    # bs = list(cars[car])[0][0].shape[0]
    bs = BATCH_SIZE
    #first calculate the total training data points across clinets
    # global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names])*bs
    global_count = sum([tf.data.experimental.cardinality(carx.train_data).numpy() for carx in cars])*bs
    # get the total number of data points held by a client
    local_count = tf.data.experimental.cardinality(car.train_data).numpy()*bs
    return local_count/global_count


def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final



def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
        
    return avg_grad

def fedavg_aggregate(cars, cars_this_round):
    scaled_local_weight_list = list()
    for car in cars_this_round:
        scaling_factor = weight_scalling_factor(cars, car)
        scaled_weights = scale_model_weights(car.weights, scaling_factor)
        scaled_local_weight_list.append(scaled_weights)
    average_weights = sum_scaled_weights(scaled_local_weight_list)
    return average_weights


class VisualizeDetections(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        visualize_detections(
            self.model, bounding_box_format="xyxy", dataset=preprocessed_test_data
        )


def default_preprocess(self, dataset):
    def batch_format_fn(element):
        """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
        return collections.OrderedDict(
            x=tf.reshape(element["pixels"], [-1, 784]),
            y=tf.reshape(element["label"], [-1, 1]),
        )

    return (
        dataset.repeat(self.local_epochs)
        .shuffle(self.shuffle_buffer, seed=1)
        .batch(self.batch_size)
        .map(batch_format_fn)
        .prefetch(self.prefetch_buffer)
    )


def default_agent_selection(self, cars):
    """Select half of the clients randomly."""
    clients = np.random.choice(cars, len(cars) // 2, replace=False)
    if len(clients) == 0:
        clients = cars
    return clients


class WirelessFedODSimulator:
    def __init__(self, num_clients=3):
        self.train_data_list = None
        self.test_data = None  # Note: pre-preprocess the test data
        self.federated_train_data = None
        self.model_fn = None
        self.agent_selection_fn = default_agent_selection
        self.preprocess_fn = default_preprocess
        self.local_epochs = 5
        self.batch_size = 20
        self.shuffle_buffer = 100
        self.prefetch_buffer = 10
        self.round_num = 0
        self.metrics = None
        self.cars = []
        self.global_weights = None
        self.num_clients = num_clients
        self.training_process = None
        self.train_state = None
        self.train_result = None
        self.sample_clients = None
        self.client_optimizer_fn = lambda: keras.optimizers.SGD(
            learning_rate=0.005, momentum=0.9, global_clipnorm=10.0
        )
        self.server_optimizer_fn = lambda: keras.optimizers.SGD(learning_rate=1.0)
        self.time_started = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.callbacks = [
            tf.keras.callbacks.TensorBoard(
                log_dir=f"logs/{self.time_started}/global",
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                write_steps_per_second=True,
            ),
            # VisualizeDetections(),
            # keras_cv.callbacks.PyCOCOCallback(self.test_data, bounding_box_format="xyxy")
        ]
        self.file_writer = tf.summary.create_file_writer(f"logs/{self.time_started}/global/eval")
        # file_writer.set_as_default()
        self.importance_fn = None

    # TODO: Handle the dataset in a more generic way
    def initialize_cars(self):
        if self.train_data_list is None:
            raise ValueError("Training data is not set.")
        if self.test_data is None:
            raise ValueError("Test data is not set.")
        if self.model_fn is None:
            raise ValueError("Model function is not set.")
        if self.preprocess_fn is None:
            raise ValueError("Preprocess function is not set.")
        if self.client_optimizer_fn is None:
            raise ValueError("Client optimizer function is not set.")

        print("Initializing cars")
        for i in range(self.num_clients):
            car = Car(
                i,
                self.model_fn,
                self.train_data_list[i],
                self.test_data,
                self.time_started,
            )
            car.preprocessed_test_data = self.test_data  # TODO: temporary interface
            car.optimizer_fn = self.client_optimizer_fn
            car.preprocess_fn = self.preprocess_fn
            car.local_epochs = self.local_epochs
            self.cars.append(car)

    # @property
    # def num_clients(self):
    #     return len(self.cars)

    def set_hyperparameters(
        self,
        num_clients=10,
        local_epochs=5,
        batch_size=20,
        shuffle_buffer=100,
        prefetch_buffer=10,
    ):
        self.num_clients = num_clients
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.prefetch_buffer = prefetch_buffer

    def initialize(self):
        self.round_num = 0
        self.global_weights = self.model_fn().get_weights()
        self.initialize_cars()

    def run_round(self):
        if self.model_fn is None:
            raise ValueError("Model function is not set.")
        if self.train_data_list is None:
            raise ValueError("Training data is not set.")
        if self.test_data is None:
            raise ValueError("Test data is not set.")
        if self.preprocess_fn is None:
            raise ValueError("Preprocess function is not set.")

        if self.global_weights is None or self.cars == []:
            self.initialize()

        self.cars_this_round = self.agent_selection_fn(self, self.cars)
        if len(self.cars_this_round) == 0:
            raise ValueError("No clients selected.")

        # print(f"Selected clients: {cars_this_round}")
        print("Selected clients:", end=" ")
        print(", ".join(str(car) for car in self.cars_this_round))

        for car in tqdm(self.cars_this_round, desc="Training cars"):
            keras.backend.clear_session()
            car.weights = self.global_weights
            car.train()

        # Update global weights with the scaled average of the local weights
        total_samples = sum(len(car.train_data) for car in self.cars_this_round)
        weighted_weights = [
            [layer * len(car.train_data) / total_samples for layer in car.weights]
            for car in self.cars_this_round
        ]
        self.global_weights = [
            np.average(weights, axis=0) for weights in zip(*weighted_weights)
        ]

        # reference implementation
        self.global_weights = fedavg_aggregate(self.cars, self.cars_this_round)

        self.round_num += 1
        self.evaluate()

    def evaluate(self):
        if self.model_fn is None:
            raise ValueError("Model function is not set.")
        if self.test_data is None:
            raise ValueError("Test data is not set.")
        if self.preprocess_fn is None:
            raise ValueError("Preprocess function is not set.")
        if self.global_weights is None:
            raise ValueError("Global weights are not set, run a round first.")

        print("Evaluating global model")
        model = self.model_fn()
        model.set_weights(self.global_weights)
        # preprocessed_test_data = self.preprocess_fn(self.test_data) # TODO: Preprocess only once
        callbacks = [
            keras_cv.callbacks.PyCOCOCallback(
                self.test_data, bounding_box_format="xyxy"
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=f"logs/{self.time_started}/global",
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                write_steps_per_second=True,
            ),
        ]
        result = model.evaluate(self.test_data, callbacks=callbacks, verbose=2, steps=self.test_data.element_spec[0].shape[0])
        # tf.summary.scalar("loss", result, step=self.round_num)
        with self.file_writer.as_default(step=self.round_num):
            tf.summary.scalar("loss", result)
        self.print_metrics(model, result)

    def visualize_detections(self):
        if self.model_fn is None:
            raise ValueError("Model function is not set.")
        if self.test_data is None:
            raise ValueError("Test data is not set.")
        # if self.preprocess_fn is None:
        #     raise ValueError("Preprocess function is not set.")
        if self.global_weights is None:
            raise ValueError("Global weights are not set, run a round first.")

        print("Visualizing detections")
        model = self.model_fn()
        model.set_weights(self.global_weights)
        visualize_detections(model, self.test_data)

    @staticmethod
    def print_metrics(model, result):
        print()
        print("Round Metrics:")
        if isinstance(result, list):
            for name, value in zip(model.metrics_names, result):
                print(f"{name}: {value:.4f}", end=", ")
        else:
            print(f"loss: {result:.4f}", end=", ")
        print(end="\n\n")

    # def make_federated_data(self, client_data, client_ids):
    #     return [
    #         self.preprocess_fn(client_data.create_tf_dataset_for_client(x))
    #         for x in client_ids
    #     ]

    # def run_epoch(self):
    #     if self.train_state is None or self.round_num is None:
    #         self.initialize()
    #     self.round_num += 1
    #     self.sample_clients = self.agent_selection_fn(self.train_data, self.num_clients)
    #     self.federated_train_data = self.make_federated_data(
    #         self.train_data, self.sample_clients
    #     )
    #     self.train_result = self.training_process.next(
    #         self.train_state, self.federated_train_data
    #     )
    #     self.train_state = self.train_result.state
    #     self.metrics = self.train_result.metrics["client_work"]["train"]
    #     # Format metrics to be one line
    #     print("round {:2d}, ".format(self.round_num), end="")
    #     print(f"num_clients: {len(self.sample_clients)}", end=", ")
    #     self.print_metrics(self.train_result)

    # def initialize(self):
    #     self.round_num = 0
    #     self.training_process = tff.learning.algorithms.build_weighted_fed_avg(
    #         self.model_fn,
    #         client_optimizer_fn=self.client_optimizer_fn,
    #         server_optimizer_fn=self.server_optimizer_fn,
    #     )
    #
    #     self.train_state = self.training_process.initialize()

