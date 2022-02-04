import tensorflow
import tensorflow.keras
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.models import Model
from localLayers import QuantizedConv2D, QuantizedDense
from localFunctions import activate


def resnet_layer(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True, conv_first=True, layersconfig=None):
    initializer = layersconfig['initializer']
    conv = QuantizedConv2D(num_filters, kernel_size, None, initializer, strides, layersconfig)

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = activate(x, activation)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = activate(x, activation)

        x = conv(x)

    return x


def resnet_v1(input_shape, depth, num_classes=10, initializer='heconstant', layersconfig=None):
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')

    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs, layersconfig=layersconfig)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample

            y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides, layersconfig=layersconfig)
            y = resnet_layer(inputs=y, num_filters=num_filters, activation=None, layersconfig=layersconfig)

            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match changed dims
                x = resnet_layer(inputs=x, num_filters=num_filters, kernel_size=1, strides=strides,
                                 activation=None, batch_normalization=False, layersconfig=layersconfig)

            x = tensorflow.keras.layers.add([x, y])
            x = activate(x, layersconfig["activation"])

        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)

    initializer = layersconfig['initializer']
    dense = QuantizedDense(num_classes, 'softmax', initializer, layersconfig)

    outputs = dense(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, depth, num_classes=10, initializer='heconstant', layersconfig=None):
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs, num_filters=num_filters_in, conv_first=True, layersconfig=layersconfig)

    num_filters_out = 1
    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2  # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x, num_filters=num_filters_in, kernel_size=1, strides=strides,
                             activation=activation, batch_normalization=batch_normalization,
                             conv_first=False, layersconfig=layersconfig)
            y = resnet_layer(inputs=y, num_filters=num_filters_in, conv_first=False, layersconfig=layersconfig)
            y = resnet_layer(inputs=y, num_filters=num_filters_out, kernel_size=1, conv_first=False, layersconfig=layersconfig)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x, num_filters=num_filters_out, kernel_size=1, strides=strides,
                                 activation=None, batch_normalization=False, layersconfig=layersconfig)
            x = tensorflow.keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)

    initializer = layersconfig['initializer']
    dense = QuantizedDense(num_classes, 'softmax', initializer, layersconfig)

    outputs = dense(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def MakeResNet(input_shape, version=1, n=3, layersconfig=None):
    # Computed depth from supplied model parameter n
    depth = 1
    if version == 1:
        depth = n * 6 + 2
    elif version == 2:
        depth = n * 9 + 2

    if version == 1:
        model = resnet_v1(input_shape=input_shape, depth=depth, layersconfig=layersconfig)
    else:
        model = resnet_v2(input_shape=input_shape, depth=depth, layersconfig=layersconfig)

    return model
