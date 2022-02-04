import utils
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as KB
from localFunctions import to_bit, to_sign, activate, fill_with_predefined
import numpy as np


def calc_scaling_factor(k, target):
    current_std = np.std(k)

    if current_std == 0:
        print("something's wrong, the standard deviation is zero!")
        return 1

    ampl = 1
    eps = 0.001
    min = 0
    max = ampl

    steps = 0
    while np.abs(current_std - target) / target > eps:
        qk = k * ampl
        current_std = np.std(qk)
        # print("error: {:.4f}".format(np.abs(current_std - target) / target))
        if current_std > target:
            max = ampl
            ampl = (max + min) / 2
        elif current_std < target:
            min = ampl
            ampl = (max + min) / 2
        steps += 1

        # if current_std == initial_std:
        #     print("something is wrong, the std is not changing")
        #     return ampl

    # print(k.shape, "relative approximation error: {:.6f} in {} steps".format(np.abs(current_std - target) / target, steps))

    return ampl


def KernelInitializer(initializer):
    if initializer == 'uniform':
        ki = tf.compat.v1.keras.initializers.RandomUniform(-0.1, 0.1)

    if initializer == 'normal':
        ki = tf.compat.v1.keras.initializers.RandomNormal(mean=0.0, stddev=0.001)

    if initializer == 'glorot':
        ki = tf.compat.v1.keras.initializers.glorot_normal()

    if initializer == 'he':
        ki = tf.compat.v1.keras.initializers.he_normal()

    return ki


def embed_pretrained_weights(bittensor_path, pretrained_bitplacement, krnlshape, index, trainable):
    # randomly filled bit_tensor
    bit_tensor_sign_magnitude = utils.predefine_nonzeroweight_bittensor(krnlshape)

    for bt, bp in zip(bittensor_path, pretrained_bitplacement):
        nbits = bp[1] - bp[0] + 1
        ptbt = utils.pretrained_bittensor(bt, index, nbits)
        bit_tensor_sign_magnitude[bp[0]:bp[1] + 1] = ptbt

    return bit_tensor_sign_magnitude


def get_weight_types(kernel):
    """
    returns the number of negative, zero and positive weights
    """
    k = KB.eval(kernel)
    neg = np.count_nonzero(k < 0)
    zeros = np.count_nonzero(k == 0)
    pos = np.count_nonzero(k > 0)

    return neg, zeros, pos


def calculate_number(signfunction, maskfunction, magnitude_block, sign_slice):
    if len(magnitude_block) == 0:
        magnitude = 1
    else:
        magnitude = 0
        for i in range(len(magnitude_block)):
            magnitude += maskfunction(magnitude_block[i]) * (2 ** i)

    # make kernel
    kernel = signfunction(sign_slice) * magnitude

    return kernel


class QuantizedConv2D(Layer):
    def __init__(self, filters, ksize, activation, initializer, stride, config, **kwargs):
        self.filters = filters
        self.ksize = ksize
        self.stride = stride
        self.initializer = initializer
        self.wbits = config["wbits"]
        self.standard_kernel = config["standard_kernel"]
        self.trainBits = config["trainableBits"]
        self.bittensor = config["pretrained_bittensor"]
        self.pretrained_bitplacement = config["pretrained_bitplacement"]
        self.inference_sequence = config["inference_sequence"]
        self.activation = activation  # ignored for convolutions, it's set outside for now

        # converts virtual bits to binary coefficients
        self.tobit = to_bit
        self.tosign = to_sign

        if stride is not None:
            self.stride = stride

        super(QuantizedConv2D, self).__init__(**kwargs)

    def build(self, input_shape):
        krnl_shape = list((self.ksize, self.ksize)) + [input_shape.as_list()[-1], self.filters]

        krnl_shape_bitwise = [self.wbits]
        krnl_shape_bitwise.extend(krnl_shape)

        print("Building Layer", self.name, krnl_shape)

        self.desired_std = np.sqrt(2 / np.prod(krnl_shape[:-1]))
        signbit = self.inference_sequence[1]
        magnitudebits = range(self.inference_sequence[0], self.inference_sequence[1])

        kernel_initializer = KernelInitializer(self.initializer)

        if self.name == "quantized_conv2d":
            index = 0
        else:
            index = int(self.name.split("_")[-1])

        if self.standard_kernel == False:

            if len(self.bittensor) > 0:
                predefined_bit_tensor = embed_pretrained_weights(self.bittensor, self.pretrained_bitplacement, krnl_shape_bitwise, index, self.trainBits)
            else:
                if self.wbits > 1:
                    predefined_bit_tensor = utils.predefine_nonzeroweight_bittensor(krnl_shape_bitwise)

            self.magnitude_block = []
            for i in magnitudebits:
                self.magnitude_block.append(self.add_weight(name='magnitudebit' + str(i) + "_Trainable" + str(self.trainBits[i]),
                                                            shape=krnl_shape,
                                                            initializer=fill_with_predefined(predefined_bit_tensor[i, ...]),
                                                            trainable=self.trainBits[i]))

            if len(self.bittensor) > 0:
                self.sign_bit = self.add_weight(name='sign_bit', shape=krnl_shape,
                                                initializer=fill_with_predefined(predefined_bit_tensor[signbit, ...]),
                                                trainable=self.trainBits[signbit])
            else:
                self.sign_bit = self.add_weight(name='sign_bit', shape=krnl_shape,
                                                initializer=kernel_initializer,
                                                trainable=self.trainBits[signbit])

            self.kernel = calculate_number(self.tosign, self.tobit, self.magnitude_block, self.sign_bit)

            kt = KB.eval(self.kernel)
            self.alpha = calc_scaling_factor(kt, self.desired_std)
            self.kernel *= self.alpha
        else:
            # uniform distribution with Kaiming He initialization technique
            a = np.sqrt(2 / np.prod(krnl_shape[:-1]))
            standard_ki = tf.compat.v1.keras.initializers.RandomUniform(-np.sqrt(12) * a / 2, np.sqrt(12) * a / 2)
            self.kernel = self.add_weight(name='kernel', shape=krnl_shape, initializer=standard_ki, trainable=True)

        super(QuantizedConv2D, self).build(input_shape)

    def call(self, x):
        y = KB.conv2d(x, self.kernel, strides=(self.stride, self.stride), padding='same')  # note we don't use biases
        return y

    def get_kernel(self):
        if self.standard_kernel:
            return KB.eval(self.kernel)
        else:
            # this calculates weights as integers, add the scaling factor to have them floats
            k = calculate_number(self.tosign, self.tobit, self.magnitude_block, self.sign_bit)
            return KB.eval(k)

    def get_bits(self):
        bittensor = []
        for i in range(len(self.magnitude_block)):
            bittensor.append(KB.eval(self.magnitude_block[i]))
        bittensor.append(KB.eval(self.sign_bit))

        return bittensor, self.alpha

    def get_nzp(self):
        return get_weight_types(KB.eval(self.kernel))


class QuantizedDense(Layer):

    def __init__(self, output_dim, activation, initializer, config, **kwargs):
        self.output_dim = output_dim
        self.initializer = initializer
        self.standard_kernel = config["standard_kernel"]
        self.trainBits = config["trainableBits"]
        self.wbits = len(self.trainBits)
        self.bittensor = config["pretrained_bittensor"]
        self.pretrained_bitplacement = config["pretrained_bitplacement"]
        self.inference_sequence = config["inference_sequence"]
        self.activation = activation

        # converts virtual bits to binary coefficients
        self.tobit = to_bit
        self.tosign = to_sign

        super(QuantizedDense, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.name == "quantized_dense":
            index = 21  # for resnet this layer will have the index 21
        else:
            index = int(self.name.split("_")[-1])

        self.kshape = (input_shape.as_list()[1], self.output_dim)

        krnl_shape = (input_shape.as_list()[1], self.output_dim)
        krnl_shape_bitwise = (self.wbits,) + krnl_shape
        print("Building Layer", self.name, krnl_shape)

        self.desired_std = np.sqrt(2 / np.prod(krnl_shape[:-1]))
        signbit = self.inference_sequence[1]
        magnitudebits = range(self.inference_sequence[0], self.inference_sequence[1])
        kernel_initializer = KernelInitializer(self.initializer)

        if self.standard_kernel == False:

            if len(self.bittensor) > 0:
                predefined_bit_tensor = embed_pretrained_weights(self.bittensor, self.pretrained_bitplacement, krnl_shape_bitwise, index, self.trainBits)
            else:
                if self.wbits > 1:
                    predefined_bit_tensor = utils.predefine_nonzeroweight_bittensor((self.wbits,) + krnl_shape)

            self.magnitude_block = []
            for i in magnitudebits:
                self.magnitude_block.append(self.add_weight(name='magnitudebit' + str(i) + "_Trainable" + str(self.trainBits[i]),
                                                            shape=krnl_shape,
                                                            initializer=fill_with_predefined(predefined_bit_tensor[i, ...]),
                                                            trainable=self.trainBits[i]))

            if len(self.bittensor) > 0:
                self.sign_bit = self.add_weight(name='sign_bit', shape=krnl_shape,
                                                initializer=fill_with_predefined(predefined_bit_tensor[signbit, ...]),
                                                trainable=self.trainBits[signbit])
            else:
                self.sign_bit = self.add_weight(name='sign_bit', shape=krnl_shape,
                                                initializer=kernel_initializer,
                                                trainable=self.trainBits[signbit])

            self.kernel = calculate_number(self.tosign, self.tobit, self.magnitude_block, self.sign_bit)

            kt = KB.eval(self.kernel)
            self.alpha = calc_scaling_factor(kt, self.desired_std)  # for good convergence
            self.kernel *= self.alpha

        else:
            # uniform distribution with Kaiming He initialization technique
            a = np.sqrt(2 / np.prod(krnl_shape[:-1]))
            standard_ki = tf.compat.v1.keras.initializers.RandomUniform(-np.sqrt(12) * a / 2, np.sqrt(12) * a / 2)
            self.kernel = self.add_weight(name='kernel', shape=krnl_shape, initializer=standard_ki, trainable=True)

        super(QuantizedDense, self).build(input_shape)

    def call(self, x):
        y = KB.dot(x, self.kernel)  # note we don't use biases
        act = activate(y, self.activation)
        return act

    def get_kernel(self):
        if self.standard_kernel:
            return KB.eval(self.kernel)
        else:
            # this calculates weights as integers, add the scaling factor to have them floats
            k = calculate_number(self.tosign, self.tobit, self.magnitude_block, self.sign_bit)
            return KB.eval(k)

    def get_bits(self):
        bittensor = []
        for i in range(len(self.magnitude_block)):
            bittensor.append(KB.eval(self.magnitude_block[i]))
        bittensor.append(KB.eval(self.sign_bit))

        return bittensor

    def get_nzp(self):
        return get_weight_types(KB.eval(self.kernel))
