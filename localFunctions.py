import tensorflow as tf
import uuid


def fill_with_predefined(src):
    def initializer(shape, dtype=None):
        return tf.Variable(src, dtype=dtype, name=uuid.uuid4().hex)

    return initializer


def activate(x, activationtype):
    if activationtype is None:
        return x

    if 'relu' in activationtype:
        return tf.keras.activations.relu(x)

    if 'softmax' in activationtype:
        return tf.keras.activations.softmax(x)

    if 'sigmoid' in activationtype:
        return tf.keras.activations.sigmoid(x)

    if 'swish' in activationtype:
        return tf.keras.activations.sigmoid(x) * x

    if "elu" in activationtype:
        return tf.keras.activations.elu(x)

    if "selu" in activationtype:
        return tf.keras.activations.selu(x)


@tf.custom_gradient
def to_bit(x):
    y = tf.sign(tf.keras.activations.relu(x))

    def grad(dy):
        return dy

    return y, grad


@tf.custom_gradient
def to_sign(x):
    y = (-1) ** to_bit(x)

    def grad(dy):
        return -dy

    return y, grad
