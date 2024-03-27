import tensorflow as tf
import tensorflow_federated as tff
import numpy as np
import collections
import random
random.seed(0)

# import the ZOD DevKit
from zod import ZodFrames

# import default constants
import zod.constants as constants
from zod.constants import Camera, Lidar, Anonymization, AnnotationProject

# import useful data classes
from zod.data_classes import LidarData

from zod.anno.object import OBJECT_CLASSES


def load_zod():
    # NOTE! Set the path to dataset and choose a version
    dataset_root = "./datasets"
    version = "mini"  # "mini" or "full"

    # initialize ZodFrames
    zod_frames = ZodFrames(dataset_root=dataset_root, version=version)

    # # get default training and validation splits
    training_frames = zod_frames.get_split(constants.TRAIN)
    validation_frames = zod_frames.get_split(constants.VAL)

    # # print the number of training and validation frames
    # print(f"Number of training frames: {len(training_frames)}")
    # print(f"Number of validation frames: {len(validation_frames)}")

    # # print out the first 5 training frames
    # print("The 5 first training frames have the ids:", sorted(list(training_frames))[:5])
    #

    def create_dataset(frames):
        # Load training_frames inta a tensorfow dataset
        image_paths = []
        bbox = []
        classes = []
        for frame_id in frames:
            frame = zod_frames[frame_id]
            # image = frame.get_image(Anonymization.DNAT)
            image_path = frame.info.get_key_camera_frame(Anonymization.DNAT).filepath
            annotations = frame.get_annotation(AnnotationProject.OBJECT_DETECTION)
            for annotation in annotations:
                if annotation.box2d:
                    image_paths.append(image_path)
                    bbox.append(annotation.box2d.xyxy)
                    classes.append(annotation.subclass)

        bbox_tensor = tf.ragged.constant(bbox)
        classes_tensor = tf.ragged.constant(classes)
        image_paths_tensor = tf.ragged.constant(image_paths)
        dataset = tf.data.Dataset.from_tensor_slices((image_paths_tensor, classes_tensor, bbox_tensor))
        return dataset

    training_dataset = create_dataset(training_frames)
    validation_dataset = create_dataset(validation_frames)

    return training_dataset, validation_dataset

class_mapping = dict(zip(range(len(OBJECT_CLASSES)), OBJECT_CLASSES))

def create_dataset(zod_frames, frame_ids):
    # Load training_frames inta a tensorfow dataset
    image_paths = []
    bbox = []
    class_ids = []
    for frame_id in frame_ids:
        frame_bboxs = []
        frame_classes = []
        frame_has_2d_bbox = False
        frame = zod_frames[frame_id]
        # image = frame.get_image(Anonymization.DNAT)
        image_path = frame.info.get_key_camera_frame(Anonymization.DNAT).filepath
        annotations = frame.get_annotation(AnnotationProject.OBJECT_DETECTION)
        for annotation in annotations:
            if annotation.box2d:
                # Only include frame in dataset if it has 2d bounding boxes
                frame_has_2d_bbox = True
                frame_bboxs.append(annotation.box2d.xyxy)
                frame_classes.append(annotation.superclass)
        if frame_has_2d_bbox:
            image_paths.append(image_path)
            bbox.append(frame_bboxs)
            # Convert classes to class_ids
            frame_class_ids = [
                list(class_mapping.keys())[list(class_mapping.values()).index(cls)]
                for cls in frame_classes
            ]
            class_ids.append(frame_class_ids)
    bbox_tensor = tf.ragged.constant(bbox)
    classes_tensor = tf.ragged.constant(class_ids)
    image_paths_tensor = tf.ragged.constant(image_paths)
    # dataset_dict = collections.OrderedDict(
    #     image_path=image_paths_tensor,
    #     classes=classes_tensor,
    #     bbox=bbox_tensor
    # )
    # dataset = tf.data.Dataset.from_tensor_slices(dataset_dict)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths_tensor, classes_tensor, bbox_tensor))
    return dataset

def load_zod2(num_clients=5, seed=0):
    client_ids = list(range(num_clients))
    def serializable_dataset_fn(client_id):
        # NOTE! Set the path to dataset and choose a version
        dataset_root = "./datasets"
        version = "mini"  # "mini" or "full"
        # initialize ZodFrames
        zod_frames = ZodFrames(dataset_root=dataset_root, version=version)
        # get default training and validation splits
        training_frames = zod_frames.get_split(constants.TRAIN)
        def get_random_sized_subset(input_list, client_id, num_clients, seed):
            random.seed(seed)
            # Generate random subset sizes
            total_elements = len(input_list)
            subset_sizes = [random.randint(1, total_elements // num_clients + 1) for _ in range(num_clients)]
            # Adjust the last subset size if the sum exceeds the list length
            while sum(subset_sizes) > total_elements:
                subset_sizes[-1] -= 1
            # Allocate subsets based on these sizes
            subsets = []
            start_index = 0
            for size in subset_sizes:
                subsets.append(input_list[start_index:start_index + size])
                start_index += size
            # Check if client_id is valid
            if client_id < 0 or client_id >= num_clients:
                raise ValueError("Invalid client_id")
            return subsets[client_id]
        client_frame_ids = get_random_sized_subset(list(training_frames), client_id, num_clients, seed)
        dataset = create_dataset(zod_frames, client_frame_ids)
        return dataset

    def serializable_dataset_fn_val(client_id):
        # NOTE! Set the path to dataset and choose a version
        dataset_root = "./datasets"
        version = "mini"  # "mini" or "full"
        # initialize ZodFrames
        zod_frames = ZodFrames(dataset_root=dataset_root, version=version)
        # get default training and validation splits
        training_frames = zod_frames.get_split(constants.VAL)
        def get_random_sized_subset(input_list, client_id, num_clients, seed):
            random.seed(seed)
            # Generate random subset sizes
            total_elements = len(input_list)
            subset_sizes = [random.randint(1, total_elements // num_clients + 1) for _ in range(num_clients)]
            # Adjust the last subset size if the sum exceeds the list length
            while sum(subset_sizes) > total_elements:
                subset_sizes[-1] -= 1
            # Allocate subsets based on these sizes
            subsets = []
            start_index = 0
            for size in subset_sizes:
                subsets.append(input_list[start_index:start_index + size])
                start_index += size
            # Check if client_id is valid
            if client_id < 0 or client_id >= num_clients:
                raise ValueError("Invalid client_id")
            return subsets[client_id]
        client_frame_ids = get_random_sized_subset(list(training_frames), client_id, num_clients, seed)
        dataset = create_dataset(zod_frames, client_frame_ids)
        return dataset

    client_data_train = tff.simulation.datasets.ClientData.from_clients_and_tf_fn(client_ids, serializable_dataset_fn)
    client_data_val = tff.simulation.datasets.ClientData.from_clients_and_tf_fn(client_ids, serializable_dataset_fn_val)
    return client_data_train, client_data_val


if __name__ == '__main__':
    load_zod2()
