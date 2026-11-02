import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Rescaling, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def build_mobilenet_model(input_shape=(224, 224, 3), num_classes=38):
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = False

    inputs = Input(shape=input_shape)
    x = Rescaling(scale=1. / 127.5, offset=-1)(inputs)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

if __name__ == "__main__":
    model = build_mobilenet_model()
    model.summary()
