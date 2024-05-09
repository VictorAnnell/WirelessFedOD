import collections
import datetime

import keras
import keras_cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from car import Car
from dataset import class_mapping, noniid_split_dataset

BATCH_SIZE = 1

def visualize_dataset(dataset, preprocess_fn, bounding_box_format="xyxy"):
    dataset = dataset.shuffle(100)
    single_image_dataset = dataset.take(1)
    images, y_true = next(iter(preprocess_fn(single_image_dataset, validation_dataset=True)))
    keras_cv.visualization.plot_bounding_box_gallery(
        images,
        value_range=(0, 255),
        bounding_box_format=bounding_box_format,
        y_true=y_true,
        scale=2,
        rows=1,
        cols=1,
        show=True,
        font_scale=0.7,
        class_mapping=class_mapping,
        legend=True,
    )
    plt.show()

def visualize_detection(model, dataset, preprocess_fn, bounding_box_format="xyxy"):
    dataset = dataset.shuffle(100)
    single_image_dataset = dataset.take(1)
    images, y_true = next(iter(preprocess_fn(single_image_dataset)))
    y_pred = model.predict(images)
    keras_cv.visualization.plot_bounding_box_gallery(
        images,
        value_range=(0, 255),
        bounding_box_format=bounding_box_format,
        y_true=y_true,
        y_pred=y_pred,
        scale=4,
        rows=1,
        cols=1,
        show=True,
        font_scale=0.7,
        class_mapping=class_mapping,
        legend=True,
    )
    plt.show()

def weight_scalling_factor(cars, car):
    #get the bs
    bs = BATCH_SIZE
    #first calculate the total training data points across clinets
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
        scaled_weights = scale_model_weights(car.local_weights, scaling_factor)
        scaled_local_weight_list.append(scaled_weights)
    average_weights = sum_scaled_weights(scaled_local_weight_list)
    return average_weights

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
    def __init__(self, num_clients=5):
        self.train_data = None
        self.test_data = None
        self.model_fn = None
        self.metrics = {}
        self.agent_selection_fn = default_agent_selection
        self.preprocess_fn = default_preprocess
        self.local_epochs = 1
        self.steps_per_local_epoch = None
        self.batch_size = 20
        self.shuffle_buffer = 100
        self.prefetch_buffer = 10
        self.round_num = 0
        self.cars = []
        self.global_weights = None
        self.num_clients = num_clients
        self.time_started = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.callbacks = [
            keras.callbacks.TensorBoard(
                log_dir=f"logs/{self.time_started}/global",
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                write_steps_per_second=True,
            ),
        ]
        self.file_writer = tf.summary.create_file_writer(f"logs/{self.time_started}/global/eval")

    # TODO: Handle the dataset in a more generic way
    def initialize_cars(self):
        if self.train_data is None:
            raise ValueError("Training data is not set.")
        if self.test_data is None:
            raise ValueError("Test data is not set.")
        if self.model_fn is None:
            raise ValueError("Model function is not set.")
        if self.preprocess_fn is None:
            raise ValueError("Preprocess function is not set.")

        train_data_splits = noniid_split_dataset(self.train_data, self.num_clients)
        print("Initializing cars")
        for i in range(self.num_clients):
            car = Car(
                i,
                self.model_fn,
                train_data_splits[i],
                self.time_started,
                local_epochs=self.local_epochs,
                steps_per_epoch=self.steps_per_local_epoch,
            )
            car.preprocess_fn = self.preprocess_fn
            self.cars.append(car)

    def initialize(self):
        self.round_num = 0
        self.global_weights = self.model_fn().get_weights()
        self.initialize_cars()

    def run_round(self):
        if self.model_fn is None:
            raise ValueError("Model function is not set.")
        if self.train_data is None:
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

        print("Selected clients:", end=" ")
        print(", ".join(str(car) for car in self.cars_this_round))

        for car in self.cars_this_round:
            keras.backend.clear_session()
            car.global_weights = self.global_weights
            car.train()

        # Update global weights with the scaled average of the local weights
        total_samples = sum(len(car.train_data) for car in self.cars_this_round)
        weighted_weights = [
            [layer * len(car.train_data) / total_samples for layer in car.local_weights]
            for car in self.cars_this_round
        ]
        self.global_weights = [
            np.average(weights, axis=0) for weights in zip(*weighted_weights)
        ]

        # reference implementation
        self.global_weights = fedavg_aggregate(self.cars, self.cars_this_round)

        self.round_num += 1
        # Communicate round number to cars
        for car in self.cars:
            car.round_num = self.round_num

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
        result = model.evaluate(self.preprocess_fn(self.test_data, validation_dataset=True), callbacks=self.callbacks)
        # Store metrics
        if isinstance(result, list):
            self.metrics = dict(zip(model.metrics_names, result))
        else:
            self.metrics = {"loss": result}
        self.print_metrics(model, result)

    def visualize_detection(self):
        if self.model_fn is None:
            raise ValueError("Model function is not set.")
        if self.test_data is None:
            raise ValueError("Test data is not set.")
        if self.preprocess_fn is None:
            raise ValueError("Preprocess function is not set.")
        if self.global_weights is None:
            print("Global weights are not set, using initial weights")
            self.initialize()

        print("Visualizing detection")
        model = self.model_fn()
        model.set_weights(self.global_weights)
        visualize_detection(model, self.test_data, self.preprocess_fn)

    def visualize_dataset(self, type="test"):
        if self.preprocess_fn is None:
            raise ValueError("Preprocess function is not set.")

        if type == "test":
            if self.test_data is None:
                raise ValueError("Test data is not set.")
            dataset = self.test_data
        elif type == "train":
            if self.train_data is None:
                raise ValueError("Train data is not set.")
            dataset = self.train_data
        else:
            raise ValueError("Invalid dataset type. Should be 'train' or 'test'")

        print(f"Visualizing {type} dataset")
        visualize_dataset(dataset, self.preprocess_fn)

    def print_metrics(self, model, result):
        print()
        print("Round Metrics:")
        with self.file_writer.as_default(step=self.round_num):
            if isinstance(result, list):
                for name, value in zip(model.metrics_names, result):
                    print(f"{name}: {value:.4f}", end=", ")
                    tf.summary.scalar(f"round_{name}", value)
            else:
                print(f"loss: {result:.4f}", end=", ")
                tf.summary.scalar("round_loss", result)
        print(end="\n\n")
