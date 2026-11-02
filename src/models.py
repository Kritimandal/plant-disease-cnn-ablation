import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Activation, Add, BatchNormalization, Conv2D, Dense, DepthwiseConv2D,
    Dropout, GlobalAveragePooling2D, MaxPooling2D, ReLU, Rescaling, SeparableConv2D,
)

from custom_cnn import build_custom_cnn
from transfer_mobilenet import build_mobilenet_model

CHANNELS = (32, 64, 128)
DENSE_UNITS = 128
DROPOUT = 0.5


def _compile(model):
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _head(x, num_classes):
    x = GlobalAveragePooling2D()(x)
    x = Dense(DENSE_UNITS, activation="relu")(x)
    x = Dropout(DROPOUT)(x)
    return Dense(num_classes, activation="softmax")(x)


def build_a0(input_shape=(224, 224, 3), num_classes=38):
    return build_custom_cnn(input_shape=input_shape, num_classes=num_classes)


def build_a1(input_shape=(224, 224, 3), num_classes=38):
    inputs = Input(shape=input_shape)
    x = Rescaling(1. / 255)(inputs)
    for filters in CHANNELS:
        x = Conv2D(filters, (3, 3), activation="relu")(x)
        x = MaxPooling2D((2, 2))(x)
    return _compile(Model(inputs, _head(x, num_classes), name="A1_gap"))


def build_a2(input_shape=(224, 224, 3), num_classes=38):
    inputs = Input(shape=input_shape)
    x = Rescaling(1. / 255)(inputs)
    for filters in CHANNELS:
        x = Conv2D(filters, (3, 3), use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D((2, 2))(x)
    return _compile(Model(inputs, _head(x, num_classes), name="A2_batchnorm"))


def build_a3(input_shape=(224, 224, 3), num_classes=38):
    inputs = Input(shape=input_shape)
    x = Rescaling(1. / 255)(inputs)

    x = Conv2D(CHANNELS[0], (3, 3), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2))(x)

    for filters in CHANNELS[1:]:
        x = SeparableConv2D(filters, (3, 3), use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D((2, 2))(x)

    return _compile(Model(inputs, _head(x, num_classes), name="A3_separable"))


def _inverted_residual(x, filters, expansion=4, stride=1):
    in_channels = x.shape[-1]
    y = Conv2D(in_channels * expansion, (1, 1), padding="same", use_bias=False)(x)
    y = BatchNormalization()(y)
    y = ReLU(max_value=6.0)(y)

    y = DepthwiseConv2D((3, 3), strides=stride, padding="same", use_bias=False)(y)
    y = BatchNormalization()(y)
    y = ReLU(max_value=6.0)(y)

    y = Conv2D(filters, (1, 1), padding="same", use_bias=False)(y)
    y = BatchNormalization()(y)

    if stride == 1 and in_channels == filters:
        y = Add()([x, y])
    return y


def build_a4(input_shape=(224, 224, 3), num_classes=38, expansion=4):
    inputs = Input(shape=input_shape)
    x = Rescaling(1. / 255)(inputs)

    x = Conv2D(CHANNELS[0], (3, 3), strides=2, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6.0)(x)

    for stage, filters in enumerate(CHANNELS):
        x = _inverted_residual(x, filters, expansion=expansion,
                               stride=1 if stage == 0 else 2)
        x = _inverted_residual(x, filters, expansion=expansion, stride=1)

    return _compile(Model(inputs, _head(x, num_classes), name="A4_inverted_residual"))


def build_b0(input_shape=(224, 224, 3), num_classes=38):
    return build_mobilenet_model(input_shape=input_shape, num_classes=num_classes)


ARMS = {
    "A0": ("original custom CNN", build_a0),
    "A1": ("A0 + global average pooling head", build_a1),
    "A2": ("A1 + batch normalization", build_a2),
    "A3": ("A2 + depthwise separable convolutions", build_a3),
    "A4": ("A3 + inverted residual blocks", build_a4),
    "B0": ("MobileNetV2 transfer learning (reference)", build_b0),
}


def build(arm, input_shape=(224, 224, 3), num_classes=38):
    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm!r}; expected one of {sorted(ARMS)}")
    return ARMS[arm][1](input_shape=input_shape, num_classes=num_classes)


if __name__ == "__main__":
    for name in sorted(ARMS):
        model = build(name, input_shape=(128, 128, 3))
        total = model.count_params()
        trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
        print(f"{name}  {ARMS[name][0]:<45} total={total:>12,}  trainable={trainable:>12,}")
