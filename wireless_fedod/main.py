import collections

import datetime
import keras
import keras_cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from dataset import OBJECT_CLASSES, class_mapping, load_zod_federated
from tqdm.auto import tqdm

from car import Car

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


def default_agent_selection(client_ids, num_clients):
    clients = np.random.choice(client_ids, num_clients // 2, replace=False)
    if len(clients) == 0:
        clients = client_ids
    return clients


class WirelessFedODSimulator:
    def __init__(self, num_clients=3):
        self.train_data_list = None
        self.test_data = None # Note: pre-preprocess the test data
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
            # keras_cv.callbacks.PyCOCOCallback(preprocessed_test_data, bounding_box_format="xyxy")
        ]

    # TODO: Handle the dataset in a more generic way
    def initialize_cars(self):
        print("Initializing cars")
        for i in range(self.num_clients):
            car = Car(i, self.model_fn, self.train_data_list[i], self.test_data, self.time_started)
            car.preprocessed_test_data = self.test_data # TODO: temporary interface
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

        cars_this_round = self.agent_selection_fn(self.cars, self.num_clients)
        if len(cars_this_round) == 0:
            raise ValueError("No clients selected.")

        # print(f"Selected clients: {cars_this_round}")
        print("Selected clients:", end=" ")
        print(", ".join(str(car) for car in cars_this_round))

        for car in tqdm(cars_this_round, desc="Training cars"):
            keras.backend.clear_session()
            car.weights = self.global_weights
            car.train()

        # Update global weights with the scaled average of the local weights
        total_samples = sum(len(car.train_data) for car in cars_this_round)
        weighted_weights = [
            [layer * len(car.train_data) / total_samples for layer in car.weights]
            for car in cars_this_round
        ]
        self.global_weights = [
            np.average(weights, axis=0) for weights in zip(*weighted_weights)
        ]

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
        result = model.evaluate(self.test_data, callbacks=self.callbacks, verbose=2)
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


if __name__ == '__main__':
    # Set preprocess_fn
    def preprocess_fn(dataset, validation_dataset=False):
        def load_image(image_path):
            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
            # image = keras.utils.load_img(image_path)
            return image

        def format_element_fn(image_path, classes, bboxes):
            # Create a dictionary with the image and bounding boxes as required by KerasCV
            image = load_image(image_path)
            bounding_boxes = {
                "classes": tf.cast(classes, dtype=tf.float32),
                "boxes": bboxes,
            }
            return {"images": tf.cast(image, tf.float32), "bounding_boxes": bounding_boxes}

        # def dict_to_tuple_fn(element):
        #     return element["images"], element["bounding_boxes"]

        def dict_to_tuple_fn(inputs):
            return inputs["images"], keras_cv.bounding_box.to_dense(
                inputs["bounding_boxes"], max_boxes=32
            )

        if validation_dataset:
            augmenters = [
                keras_cv.layers.Resizing(640, 640, pad_to_aspect_ratio=True, bounding_box_format="xyxy")
            ]
        else:
            augmenters = [
                keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format="xyxy"),
                keras_cv.layers.JitteredResize(
                    target_size=(640, 640), scale_factor=(0.75, 1.3), bounding_box_format="xyxy"
                ),
            ]

        dataset = dataset.map(format_element_fn, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(BATCH_SIZE * 4)
        dataset = dataset.ragged_batch(BATCH_SIZE, drop_remainder=True)

        # dataset = dataset.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)
        def create_augmenter_fn(augmenters):
            def augmenter_fn(inputs):
                for augmenter in augmenters:
                    inputs = augmenter(inputs)
                return inputs

            return augmenter_fn

        augmenter_fn = create_augmenter_fn(augmenters)
        dataset = dataset.map(augmenter_fn, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(dict_to_tuple_fn, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.repeat(2)  # Repeat for more epochs TODO: set this correctly
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset


    simulator = WirelessFedODSimulator()
    simulator.preprocess_fn = preprocess_fn

    # Load ZOD dataset
    zod_train, zod_test = load_zod_federated(num_clients=simulator.num_clients, version='full', upper_bound=99)
    # example_dataset = zod_train.create_tf_dataset_for_client(zod_train.client_ids[0])
    # preprocessed_example_dataset = simulator.preprocess_fn(example_dataset)

    # Create model_fn
    def model_fn():
        # model = keras_cv.models.RetinaNet.from_preset(
        #     "resnet50",
        #     bounding_box_format="xyxy",
        #     num_classes=len(OBJECT_CLASSES),
        # )
        # model.compile(
        #     classification_loss="focal",
        #     box_loss="smoothl1",
        #     metrics=['accuracy'],
        #     optimizer=keras.optimizers.SGD(global_clipnorm=10.0),
        #     jit_compile="auto",
        # )
        # model = keras_cv.models.YOLOV8Detector.from_preset(
        #     "resnet50_imagenet",
        #     # "yolo_v8_m_pascalvoc",
        #     # For more info on supported bounding box formats, visit
        #     # https://keras.io/api/keras_cv/bounding_box/
        #     bounding_box_format="xyxy",
        #     num_classes=len(OBJECT_CLASSES),
        # )
        # model = keras_cv.models.YOLOV8Backbone.from_preset(
        #     "yolo_v8_xs_backbone_coco",
        #     load_weights=False,
        # )
        model = keras_cv.models.YOLOV8Detector(
            num_classes=len(OBJECT_CLASSES),
            bounding_box_format="xyxy",
            backbone=keras_cv.models.YOLOV8Backbone.from_preset(
                # "yolo_v8_xs_backbone_coco"
                "yolo_v8_xs_backbone"
            ),
            fpn_depth=2
        )
        base_lr = 0.005
        # including a global_clipnorm is extremely important in object detection tasks
        optimizer = keras.optimizers.SGD(
            learning_rate=base_lr, momentum=0.9, global_clipnorm=10.0
        )
        model.compile(
            classification_loss="binary_crossentropy",
            box_loss="ciou",
            # loss=keras_cv.losses.CIoULoss(bounding_box_format="xyxy"),
            # loss='mse',
            optimizer=optimizer,
            # metrics=['accuracy']
        )
        # return keras_cv.models.YOLOV8Detector.from_preset(
        #     "yolo_v8_m_pascalvoc", bounding_box_format="xyxy"
        # )
        return model


    # Set dataset, model_fn, and agent_selection_fn
    simulator.train_data_list = zod_train
    # simulator.test_data = zod_test
    simulator.test_data = preprocess_fn(zod_test, validation_dataset=True)
    simulator.model_fn = model_fn
    # simulator.agent_selection_fn = lambda x, y: np.random.choice(x.client_ids, y, replace=False)
    # Set optimizers
    # simulator.client_optimizer_fn = lambda: keras.optimizers.Adam(learning_rate=0.001)
    # simulator.server_optimizer_fn = lambda: keras.optimizers.Adam(learning_rate=0.1)
    # Run training
    # for _ in range(5):
    #     simulator.run_round()
    simulator.run_round()
