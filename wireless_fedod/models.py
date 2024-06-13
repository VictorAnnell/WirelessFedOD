import keras
import keras_cv

yolo_prediction_decoder = keras_cv.layers.NonMaxSuppression(
    bounding_box_format="xyxy",
    from_logits=True,
    # Decrease the required threshold to make predictions get pruned out
    iou_threshold=0.6,
    # Tune confidence threshold for predictions to pass NMS
    confidence_threshold=0.51,
)


def yolov8xs_model_fn() -> keras.Model:
    from wireless_fedod.config import OBJECT_CLASSES

    model = keras_cv.models.YOLOV8Detector.from_preset(
        "yolo_v8_xs_backbone",
        num_classes=len(OBJECT_CLASSES),
        bounding_box_format="xyxy",
        fpn_depth=2,
        prediction_decoder=yolo_prediction_decoder,
    )
    optimizer = keras.optimizers.Adam(learning_rate=0.005, global_clipnorm=10.0)
    model.compile(
        classification_loss="binary_crossentropy",
        box_loss="ciou",
        optimizer=optimizer,
        jit_compile=False,
    )
    return model


def yolov8xs_coco_model_fn() -> keras.Model:
    from wireless_fedod.config import OBJECT_CLASSES

    base_model = keras_cv.models.YOLOV8Detector.from_preset(
        "yolo_v8_xs_backbone_coco",
        num_classes=len(OBJECT_CLASSES),
        bounding_box_format="xyxy",
        fpn_depth=2,
        prediction_decoder=yolo_prediction_decoder,
    )
    optimizer = keras.optimizers.Adam(learning_rate=0.005, global_clipnorm=10.0)
    base_model.compile(
        classification_loss="binary_crossentropy",
        box_loss="ciou",
        optimizer=optimizer,
        jit_compile=False,
    )
    return base_model


def yolov8s_coco_model_fn() -> keras.Model:
    from wireless_fedod.config import OBJECT_CLASSES

    model = keras_cv.models.YOLOV8Detector.from_preset(
        "yolo_v8_s_backbone_coco",
        num_classes=len(OBJECT_CLASSES),
        bounding_box_format="xyxy",
        fpn_depth=1,
        prediction_decoder=yolo_prediction_decoder,
    )
    optimizer = keras.optimizers.Adam(learning_rate=0.005, global_clipnorm=10.0)
    model.compile(
        classification_loss="binary_crossentropy",
        box_loss="ciou",
        optimizer=optimizer,
        jit_compile=False,
    )
    return model


def retinanet_resnet50_model_fn() -> keras.Model:
    from wireless_fedod.config import OBJECT_CLASSES

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
    from wireless_fedod.config import OBJECT_CLASSES

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
