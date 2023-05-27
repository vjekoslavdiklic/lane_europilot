from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def get_model_m1(width=128, height=128, depth=64):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))
    x0 = layers.Reshape((341, 562, 3,))(inputs)
    x1 = layers.Resizing(height=600, width=600)(x0)
    x2 = layers.Conv2D(25, kernel_size=(5, 5),activation='relu',strides=(2, 2))(x1)
    x3 = layers.BatchNormalization()(x2)
    x4 = layers.Conv2D(50, kernel_size=(5, 5), activation='relu',strides=(2, 2))(x3)
    x5 = layers.BatchNormalization()(x4)
    x6 = layers.Conv2D(75, kernel_size=(5, 5),activation='relu',strides=(2, 2))(x5)
    x7 = layers.BatchNormalization()(x6)
    x8 = layers.Conv2D(120, kernel_size=(3, 3), activation='relu',strides=(1, 1))(x7)
    x9 = layers.BatchNormalization()(x8)

    x10 = layers.MaxPooling2D(pool_size=(3, 3))(x9)
    x11 = layers.Flatten()(x9)

    x12 = layers.Dense(units=200, activation="relu")(x11)
    x12a = layers.Dense(units=400, activation="relu")(x12)
    x13 = tf.keras.layers.Dropout(0.3)(x12a)
    x14 = layers.Dense(units=24000, activation="sigmoid")(x13)
    x15 = layers.Reshape((120, 200, 1))(x14)
    x15aa = layers.Resizing(height=4*5*120, width=4*5*200)(x15)
    x15a = layers.MaxPooling2D(pool_size=(5, 5))(x15aa)
    x15c = layers.MaxPooling2D(pool_size=(4, 4))(x15a)
    #x15cd = layers.MaxPooling2D(pool_size=(2, 2))(x15c)
    x16 = layers.Resizing(height=341,width=562)(x15c)
    #x17 = layers.BatchNormalization()(x16)
    x18= layers.ThresholdedReLU(theta=0.001)(x16)

    model = keras.Model(inputs, x18, name="m1")
    return model


def get_model_m2(width=128, height=128, depth=64):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((None, None,3))
    #x0 = layers.Reshape((341, 562, 3,))(inputs)
    x1 = layers.Resizing(height=500, width=500)(inputs)
    x2 = layers.Conv2D(25, kernel_size=(5, 5),activation='relu',strides=(2, 2))(x1)
    x3 = layers.BatchNormalization()(x2)
    x4 = layers.Conv2D(50, kernel_size=(5, 5), activation='relu',strides=(2, 2))(x3)
    x5 = layers.BatchNormalization()(x4)
    x6 = layers.Conv2D(75, kernel_size=(5, 5),activation='relu',strides=(2, 2))(x5)
    x7 = layers.BatchNormalization()(x6)
    x8 = layers.Conv2D(120, kernel_size=(3, 3), activation='relu',strides=(1, 1))(x7)
    x9 = layers.BatchNormalization()(x8)
    x10 = layers.MaxPooling2D(pool_size=(3, 3))(x9)
    x11 = layers.Flatten()(x10)
    #x11a = layers.Reshape((-1,1))(x11)
    #x11a = layers.Flatten()(x11)

    x12 = layers.Dense(units=100, activation="relu")(x11)
    x13 = tf.keras.layers.Dropout(0.2)(x12)
    x13a = layers.Dense(units=50, activation="relu")(x13)
    x13b = tf.keras.layers.Dropout(0.2)(x13a)
    x14 = layers.Dense(units=40000, activation="sigmoid")(x13b)
    x15 = layers.Reshape((200, 200, 1))(x14)
    ##x15aa = layers.Resizing(height=120, width=200)(x15)
    #x15aa = layers.Resizing(height=4*5*120, width=4*5*200)(x15)
    #x15a = layers.MaxPooling2D(pool_size=(5, 5))(x15aa)
    #x15c = layers.MaxPooling2D(pool_size=(4, 4))(x15a)
    #x15cd = layers.MaxPooling2D(pool_size=(2, 2))(x15c)
    x16 = layers.Resizing(height=200,width=200)(x15)
    #x17 = layers.BatchNormalization()(x16)
    x18= layers.ThresholdedReLU(theta=0.01)(x16)

    model = keras.Model(inputs, x18, name="myexpnet")
    return model


def get_model_unet(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)
    #outputstrashholded=layers.ThresholdedReLU(theta=0.1)(outputs)
    # Define the model
    #model = keras.Model(inputs, outputstrashholded)
    model = keras.Model(inputs, outputs)
    return model
