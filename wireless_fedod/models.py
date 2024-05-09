import keras
import keras_cv
from dataset import OBJECT_CLASSES


def yolov8_model_fn() -> keras.Model:
    model = keras_cv.models.YOLOV8Detector(
        num_classes=len(OBJECT_CLASSES),
        bounding_box_format="xyxy",
        backbone=keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_xs_backbone"),
        fpn_depth=2,
    )
    base_lr = 0.005
    # including a global_clipnorm is extremely important in object detection tasks
    optimizer = keras.optimizers.SGD(learning_rate=base_lr, momentum=0.9, global_clipnorm=10.0)
    model.compile(
        classification_loss="binary_crossentropy",
        box_loss="ciou",
        optimizer=optimizer,
    )
    return model
