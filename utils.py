import os
import uuid
import pickle

import numpy as np
from zipfile import ZipFile

import matplotlib.pyplot as plt


def SplitToCompactGrid(n):
    cols = int(np.ceil(np.sqrt(n)))
    lines = int(n / cols)

    if n % cols != 0:
        lines += 1

    return lines, cols


def copyfilesto(mypath, mainuuid):
    if not os.path.exists(mypath):
        os.makedirs(mypath)

    zipObj = ZipFile(mypath + 'code_' + mainuuid + '.zip', 'w')

    zipObj.write('localFunctions.py')
    zipObj.write('localLayers.py')
    zipObj.write('ResNetBuilder.py')
    zipObj.write('Trainer.py')
    zipObj.write('utils.py')
    zipObj.close()


def make_outputpath(config):
    RunID = uuid.uuid4().hex

    basedir = config["basedir"]
    mypath = basedir + config['name']
    mypath += "/" + config['initializer']
    mypath += "/" + RunID[-7:] + "/"

    return mypath


def getkernels(net):
    weights = []

    for l in range(1, len(net.layers)):
        if "quantized" in net.layers[l].name:
            w = net.layers[l].get_kernel()
            weights.append(w)

    return weights


def getbits(net):
    bits = []

    for l in range(1, len(net.layers)):
        if "quantized" in net.layers[l].name:
            bittensor = net.layers[l].get_bits()
            bits.append(bittensor)

    return bits


def getNZP(net):
    nsum = 0
    zsum = 0
    psum = 0

    for l in range(1, len(net.layers)):
        if "quantized" not in net.layers[l].name:
            continue

        neg, zero, pos = net.layers[l].get_nzp()
        nsum += neg
        zsum += zero
        psum += pos

        # print(net.layers[l].name, neg, zero, pos)

    return nsum, zsum, psum


def plot_uniques(X, output=None, show=None):
    figsize = (40, 20)
    fontsizes = 25

    if len(X) < 4:
        fig, axes = plt.subplots(1, len(X), figsize=figsize, dpi=60)
        all_axes = axes.reshape(-1)
    else:
        nx, ny = SplitToCompactGrid(len(X))
        fig, axes = plt.subplots(nx, ny, figsize=figsize, dpi=60)
        axes = axes.reshape(-1)
        all_axes = axes

    for i, x in enumerate(X):
        newx = x.reshape(-1)
        axes[i].hist(newx, bins=120)
        # axes[i].set_xlim((-2.1, 2.1))
        # axes[i].set_title("layer" + str(i), fontsize=fontsizes+10)
        axes[i].grid(True)
        # axes[i].set_yscale('log')
        # fig.savefig("weights.pdf")

    i = 0
    for a in all_axes:

        for tick in a.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsizes)

        for tick in a.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsizes)

        a.yaxis.set_major_locator(plt.MaxNLocator(5))
        left, right = a.get_xlim()
        a.set_xlim((left, right))
        # a.set_ylim((1, 10 ** 4.2))
        a.xaxis.set_major_locator(plt.MaxNLocator(3))
        all_axes[0].set_ylabel("Frequency", fontsize=fontsizes)
        a.set_xlabel("Layer" + str(i), fontsize=fontsizes)
        i += 1

        # a.set_xticks([left,0,right])

    plt.tight_layout(pad=1)
    if output is not None:
        fig.savefig(output)

    if show:
        plt.show()
        plt.close()


def predefine_nonzeroweight_bittensor(shape):
    nlp = np.prod(shape[:-1])
    a = np.sqrt(2 / nlp)
    nbits = shape[0]
    distribution = a * np.random.normal(0, 1, shape)

    # mlp here
    if len(shape) == 3:
        for i in range(shape[1]):
            for j in range(shape[2]):
                while np.all(distribution[:-1, i, j] <= 0):
                    distribution[:-1, i, j] = a * np.random.normal(0, 1, nbits - 1)

    # kernel here
    if len(shape) == 5:
        for in_channels in range(shape[3]):
            for out_channels in range(shape[4]):
                for i in range(shape[1]):
                    for j in range(shape[2]):
                        while np.all(distribution[:-1, i, j, in_channels, out_channels] <= 0):
                            distribution[:-1, i, j, in_channels, out_channels] = a * np.random.normal(0, 1, nbits - 1)

    return distribution


def kernel_to_bit_tensor(kernel, nbits):
    get_bin = lambda x, n: format(x, 'b').zfill(n)
    nlp = np.prod(kernel.shape[:-1])
    a = np.sqrt(2 / nlp)
    bit_tensor = None

    bits = np.zeros(nbits - 1)
    # mlp weights
    if len(kernel.shape) == 2:
        bit_tensor = np.zeros((nbits,) + kernel.shape)
        for lins in range(kernel.shape[0]):
            for cols in range(kernel.shape[1]):
                k = kernel[lins, cols]
                binary_form = get_bin(np.abs(k), nbits - 1)
                binary_form = binary_form[::-1]

                for i in range(0, nbits - 1):
                    bit = int(binary_form[i])
                    if bit == 0:
                        bit_tensor[i, lins, cols] = -a * np.abs(np.random.normal(0, 1))
                    else:
                        bit_tensor[i, lins, cols] = a * np.abs(np.random.normal(0, 1))

                # set the sign bit
                if k > 0:
                    bit_tensor[nbits - 1, lins, cols] = a * np.abs(np.random.normal(0, 1))
                elif k < 0:
                    bit_tensor[nbits - 1, lins, cols] = -a * np.abs(np.random.normal(0, 1))
                else:
                    bit_tensor[nbits - 1, lins, cols] = a * np.random.normal(0, 1)

    if len(kernel.shape) == 4:
        bit_tensor = np.zeros((nbits,) + kernel.shape)

        for in_channels in range(kernel.shape[2]):
            for out_channels in range(kernel.shape[3]):
                for lins in range(kernel.shape[0]):
                    for cols in range(kernel.shape[1]):
                        k = kernel[lins, cols, in_channels, out_channels]
                        binary_form = get_bin(np.abs(k), nbits - 1)
                        binary_form = binary_form[::-1]

                        for i in range(0, nbits - 1):
                            bit = int(binary_form[i])
                            if bit == 0:
                                bit_tensor[i, lins, cols, in_channels, out_channels] = -a * np.abs(np.random.normal(0, 1))
                            else:
                                bit_tensor[i, lins, cols, in_channels, out_channels] = a * np.abs(np.random.normal(0, 1))

                        # set the sign bit
                        if k > 0:
                            bit_tensor[nbits - 1, lins, cols, in_channels, out_channels] = a * np.abs(np.random.normal(0, 1))
                        elif k < 0:
                            bit_tensor[nbits - 1, lins, cols, in_channels, out_channels] = -a * np.abs(np.random.normal(0, 1))
                        else:
                            bit_tensor[nbits - 1, lins, cols, in_channels, out_channels] = a * np.random.normal(0, 1)

    return bit_tensor


def pretrained_bittensor(mypath, index, nbits):
    weights = pickle.load(open(mypath[0], "rb"))

    # weights are floats
    if mypath[1] == "f":
        kernel_as_integer = np.round(weights[index] / (np.min(np.abs(weights[index][np.abs(weights[index] > 0)])))).astype(np.int)
        bit_tensor_from_kernel = kernel_to_bit_tensor(kernel_as_integer, nbits)
        return bit_tensor_from_kernel

    # weights are integers
    if mypath[1] == "i":
        kernel_as_integer = weights[index]
        bit_tensor_from_kernel = kernel_to_bit_tensor(kernel_as_integer, nbits)
        return bit_tensor_from_kernel

    # weights are actually raw bits
    if mypath[1] == "b":
        return weights[index]
