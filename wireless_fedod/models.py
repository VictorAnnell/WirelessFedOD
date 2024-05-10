import keras
import keras_cv


def yolov8_model_fn() -> keras.Model:
    from config import OBJECT_CLASSES

    model = keras_cv.models.YOLOV8Detector(
        num_classes=len(OBJECT_CLASSES),
        bounding_box_format="xyxy",
        backbone=keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_xs_backbone", load_weights=False),
        fpn_depth=2,
    )
    base_lr = 0.005
    # including a global_clipnorm is extremely important in object detection tasks
    optimizer = keras.optimizers.SGD(learning_rate=base_lr, momentum=0.9, global_clipnorm=10.0)
    model.compile(
        classification_loss="binary_crossentropy",
        box_loss="ciou",
        optimizer=optimizer,
        jit_compile=False,
    )
    return model


def retinanet_resnet50_model_fn() -> keras.Model:
    from config import OBJECT_CLASSES

    model = keras_cv.models.RetinaNet(
        num_classes=len(OBJECT_CLASSES),
        bounding_box_format="xyxy",
        backbone=keras_cv.models.ResNetBackbone.from_preset("resnet50", load_weights=False),
    )
    optimizer = keras.optimizers.SGD(learning_rate=0.005, momentum=0.9, global_clipnorm=10.0)
    model.compile(
        classification_loss="focal",
        box_loss="smoothl1",
        optimizer=optimizer,
        jit_compile=False,
    )
    return model


def retinanet_resnet50_imagenet_model_fn() -> keras.Model:
    from config import OBJECT_CLASSES

    model = keras_cv.models.RetinaNet(
        num_classes=len(OBJECT_CLASSES),
        bounding_box_format="xyxy",
        backbone=keras_cv.models.ResNetBackbone.from_preset("resnet50_imagenet", load_weights=False),
    )
    optimizer = keras.optimizers.SGD(learning_rate=0.005, momentum=0.9, global_clipnorm=10.0)
    model.compile(
        classification_loss="focal",
        box_loss="smoothl1",
        optimizer=optimizer,
        jit_compile=False,
    )
    return model
