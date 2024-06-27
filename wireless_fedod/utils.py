import keras
import keras_cv
import matplotlib.pyplot as plt
import tensorflow as tf

class EvaluateCOCOMetricsCallback(keras.callbacks.Callback):
    def __init__(self, data, save_path):
        super().__init__()
        self.data = data
        self.metrics = keras_cv.metrics.BoxCOCOMetrics(
            bounding_box_format="xyxy",
            evaluate_freq=1e9,
        )

        self.save_path = save_path
        self.best_map = -1.0

    def on_epoch_end(self, epoch, logs):
        self.metrics.reset_state()
        for batch in self.data:
            images, y_true = batch[0], batch[1]
            y_pred = self.model.predict(images, verbose=0)
            self.metrics.update_state(y_true, y_pred)

        metrics = self.metrics.result(force=True)
        logs.update(metrics)

#        current_map = metrics["MaP"]
#        if current_map > self.best_map:
#            self.best_map = current_map
#            self.model.save(self.save_path)  # Save the model when mAP improves

        return logs

    def on_test_end(self, logs):
        self.metrics.reset_state()
        for batch in self.data:
            images, y_true = batch[0], batch[1]
            y_pred = self.model.predict(images, verbose=0)
            self.metrics.update_state(y_true, y_pred)

        metrics = self.metrics.result(force=True)
        logs.update(metrics)

#        current_map = metrics["MaP"]
#        if current_map > self.best_map:
#            self.best_map = current_map
#            self.model.save(self.save_path)  # Save the model when mAP improves

        return logs

def load_image(image_path):
    image = tf.io.read_file(tf.cast(image_path, dtype=tf.string))
    image = tf.image.decode_jpeg(image, channels=3)
    return image


def format_element_fn(image_path, classes, bboxes):
    # Create a dictionary with the image and bounding boxes as required by KerasCV
    image = load_image(image_path)
    bounding_boxes = {
        "classes": tf.cast(classes, dtype=tf.float32),
        "boxes": tf.cast(bboxes, dtype=tf.float32),
    }
    return {
        "images": tf.cast(image, tf.float32),
        "bounding_boxes": keras_cv.bounding_box.to_dense(bounding_boxes),
        #"bounding_boxes": bounding_boxes,
    }


def dict_to_tuple_fn(inputs):
    #return inputs["images"], keras_cv.bounding_box.to_dense(inputs["bounding_boxes"], max_boxes=500)
    return inputs["images"], inputs["bounding_boxes"]


def visualize_dataset(dataset, preprocess_fn, class_mapping=None, bounding_box_format="xyxy"):
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


def visualize_detection(model, dataset, preprocess_fn, class_mapping=None, bounding_box_format="xyxy"):
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
    # get the bs
    from wireless_fedod.dataset import BATCH_SIZE  # TODO: remove this when batch size is configurable

    bs = BATCH_SIZE
    # first calculate the total training data points across clinets
    global_count = sum([tf.data.experimental.cardinality(carx.train_data).numpy() for carx in cars]) * bs
    # get the total number of data points held by a client
    local_count = tf.data.experimental.cardinality(car.train_data).numpy() * bs
    return local_count / global_count


def scale_model_weights(weight, scalar):
    """function for scaling a models weights"""
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final


def sum_scaled_weights(scaled_weight_list):
    """Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights"""
    avg_grad = list()
    # get the average grad accross all client gradients
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
