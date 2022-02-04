import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import utils, ResNetBuilder
import time, uuid, pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as KB
from shutil import copy2

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MaxPooling2D, Flatten
from localLayers import QuantizedConv2D, QuantizedDense
from localFunctions import activate

np.set_printoptions(edgeitems=3, linewidth=256)


def get_session():
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    return tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


tf.compat.v1.keras.backend.set_session(get_session())

mainUUID = uuid.uuid4().hex[-7:]


def mnist():
    (Xtrain, Ytrain), (Xtest, Ytest) = tf.keras.datasets.mnist.load_data()
    Xtrain = Xtrain - np.mean(Xtrain, axis=0)
    Xtest = Xtest - np.mean(Xtest, axis=0)

    Xtrain /= (np.std(Xtrain))
    Xtest /= (np.std(Xtest))

    Xtrain = Xtrain.reshape(-1, 28 * 28)
    Xtest = Xtest.reshape(-1, 28 * 28)

    TrainLabels = np.zeros((len(Ytrain), 10))
    for i in range(0, len(Ytrain)):
        TrainLabels[i, Ytrain[i]] = 1

    TestLabels = np.zeros((len(Ytest), 10))
    for i in range(0, len(Ytest)):
        TestLabels[i, Ytest[i]] = 1

    return Xtrain, TrainLabels, Xtest, TestLabels, 10


def cifar10(ss=None):
    (xtrain, ytrain), (xtest, ytest) = tf.keras.datasets.cifar10.load_data()

    TrainInput = xtrain / 255
    TestInput = xtest / 255

    TrainInput -= np.mean(TrainInput, axis=0)
    TestInput -= np.mean(TestInput, axis=0)

    TrainInput /= (np.std(TrainInput))
    TestInput /= (np.std(TestInput))

    TrainLabels = np.zeros((len(ytrain), 10))
    for i in range(0, len(ytrain)):
        TrainLabels[i, ytrain[i]] = 1

    TestLabels = np.zeros((len(ytest), 10))
    for i in range(0, len(ytest)):
        TestLabels[i, ytest[i]] = 1

    Xtrain, Ytrain, Xtest, Ytest, nclasses = np.ascontiguousarray(TrainInput), np.ascontiguousarray(TrainLabels), np.ascontiguousarray(TestInput), np.ascontiguousarray(
        TestLabels), 10

    data = Xtrain[:ss], Ytrain[:ss], Xtest[:ss], Ytest[:ss], nclasses
    return data


def training_schedule(schedule):
    schedule_epochs = schedule[0]
    schedule_lrates = schedule[1]

    expanded_lrates = []
    for i in range(len(schedule_epochs)):
        for j in range(schedule_epochs[i]):
            expanded_lrates.append(schedule_lrates[i])

    return expanded_lrates


def build_LeNet300(config):
    """
    Hardcoded LeNet, could be moved somewhere else and generalized
    """

    initializer = config["initializer"]
    act = config["activation"]
    input_img = Input(shape=(28 * 28,))
    L300 = QuantizedDense(300, act, initializer, config)(input_img)
    L100 = QuantizedDense(100, act, initializer, config)(L300)
    L10 = QuantizedDense(10, "softmax", initializer, config)(L100)

    model = Model(input_img, L10)
    model._name = "LeNet300" + "_ID" + uuid.uuid4().hex[-7:]

    return model


def build_conv6(input_shape, layersconfig):
    """
    Hardcoded Conv6, could be moved somewhere else and generalized
    """
    inputs = Input(shape=input_shape)
    activation = layersconfig["activation"]

    conv1_1 = activate(QuantizedConv2D(64, 3, None, "he", 1, layersconfig)(inputs), activation)
    conv1_2 = activate(QuantizedConv2D(64, 3, None, "he", 1, layersconfig)(conv1_1), activation)
    pool1 = MaxPooling2D(pool_size=(2, 2), name="pool1")(conv1_2)

    conv1_3 = activate(QuantizedConv2D(128, 3, None, "he", 1, layersconfig)(pool1), activation)
    conv1_4 = activate(QuantizedConv2D(128, 3, None, "he", 1, layersconfig)(conv1_3), activation)
    pool2 = MaxPooling2D(pool_size=(2, 2), name="pool2")(conv1_4)

    conv1_5 = activate(QuantizedConv2D(256, 3, None, "he", 1, layersconfig)(pool2), activation)
    conv1_6 = activate(QuantizedConv2D(256, 3, None, "he", 1, layersconfig)(conv1_5), activation)
    pool3 = MaxPooling2D(pool_size=(2, 2), name="pool3")(conv1_6)

    flat = Flatten()(pool3)

    # Add fully connected layers.
    fc1 = QuantizedDense(256, "relu", "he", layersconfig, name="quantized_dense_6")(flat)
    fc2 = QuantizedDense(256, "relu", "he", layersconfig, name="quantized_dense_7")(fc1)
    fc3 = QuantizedDense(10, "softmax", "he", layersconfig, name="quantized_dense_8")(fc2)

    return Model(inputs=inputs, outputs=[fc3], name="conv6")


def CIFAR_Trainer(network, data, mypath, config):
    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    Xtrain, Ytrain, Xtest, Ytest, nclasses = data
    datagen.fit(Xtrain)

    print("\nEvaluate network with no training:")
    TrainL0, TrainA0 = network.evaluate(Xtrain, Ytrain, batch_size=200, verbose=2)
    TestL0, TestA0 = network.evaluate(Xtest, Ytest, batch_size=200, verbose=2)

    neg, zero, pos = utils.getNZP(network)
    NZPMasks = [[neg, zero, pos]]
    denom = (neg + zero + pos)
    print("neg: {}, zero: {}, pos: {} - {:.7f}, {:.7f}, {:.7f}".format(neg, zero, pos, neg / denom, zero / denom, pos / denom))

    TrainLoss = np.asarray([TrainL0])
    TrainAccuracy = np.asarray([TrainA0])

    TestLoss = np.asarray([TestL0])
    TestAccuracy = np.asarray([TestA0])

    maxtrainacc = TrainA0
    maxtestacc = TestA0

    loss, metric = network.metrics_names
    weights = utils.getkernels(network)
    file = open(mypath + "Weights0.pkl", "wb")
    pickle.dump(weights, file)
    file.close()

    expanded_lrates = training_schedule(config["lr_schedule"])
    batchsize = config["batchsize"]
    maxepochs = len(expanded_lrates)

    epoch = 0
    utils.plot_uniques(weights, mypath + "weights" + str(epoch) + ".png")
    while epoch < maxepochs:
        start_time = time.time()
        lr = expanded_lrates[epoch]
        KB.set_value(network.optimizer.lr, lr)

        if config["verbose"] == 1:
            weights = utils.getkernels(network)
            utils.plot_uniques(weights, mypath + "weights" + str(epoch) + ".png")

        fit_history = network.fit_generator(datagen.flow(Xtrain, Ytrain, batch_size=batchsize), validation_data=(Xtest, Ytest), epochs=1, verbose=0, shuffle=True)

        TrainLoss = np.append(TrainLoss, fit_history.history[loss])
        TestLoss = np.append(TestLoss, fit_history.history['val_loss'])

        TrainAccuracy = np.append(TrainAccuracy, fit_history.history[metric])
        TestAccuracy = np.append(TestAccuracy, fit_history.history['val_' + metric])

        if TestAccuracy[-1] > maxtestacc:
            predictions = network.predict(Xtest)
            np.save(mypath + "BestTestPredictions.npy", predictions)
            weights = utils.getkernels(network)
            file = open(mypath + "BestWeights.pkl", "wb")
            pickle.dump(weights, file)
            file.close()

        maxtrainacc = max(maxtrainacc, TrainAccuracy[-1])
        maxtestacc = max(maxtestacc, TestAccuracy[-1])

        print("trainable bits  - {}".format(''.join(str(e) for e in config["trainableBits"])), len(config["trainableBits"]))
        print("learn rate      - {:.13f}".format(lr))
        print("trn             - loss: {:.7f} | acc {:.4f} | best: {:.4f} | avg 5: {:.4f}".format(TrainLoss[-1], TrainAccuracy[-1], maxtrainacc, np.mean(TrainAccuracy[-5:])))
        print("tst             - loss: {:.7f} | acc {:.4f} | best: {:.4f} | avg 5: {:.4f}".format(TestLoss[-1], TestAccuracy[-1], maxtestacc, np.mean(TestAccuracy[-5:])))
        neg, zero, pos = utils.getNZP(network)
        denom = (neg + zero + pos)
        NZPMasks.append([neg, zero, pos])
        print("nzp             - {} | {} | {}  -  {:.7f}, {:.7f}, {:.7f}".format(neg, zero, pos, neg / denom, zero / denom, pos / denom))
        print("epoch           - {}/{},  runtime: {:.3f} seconds".format(epoch + 1, maxepochs, time.time() - start_time))
        epoch += 1

    Logs = {"trainLoss": TrainLoss,
            "testLoss": TestLoss,
            "trainAccuracy": TrainAccuracy,
            "testAccuracy": TestAccuracy,
            "neg_zero_pos_masks": NZPMasks
            }

    np.savetxt(mypath + 'TrainAccuracy.txt', TrainAccuracy, delimiter=',')
    np.savetxt(mypath + 'TestAccuracy.txt', TestAccuracy, delimiter=',')

    predictions = network.predict(Xtest)
    np.save(mypath + "TestPredictions.npy", predictions)

    file = open(mypath + "TrainLogs.pkl", "wb")
    pickle.dump(Logs, file)
    file.close()

    weights = utils.getkernels(network)
    utils.plot_uniques(weights, mypath + "weights" + str(epoch) + ".png")

    file = open(mypath + "Weights" + str(epoch) + ".pkl", "wb")
    pickle.dump(weights, file)
    file.close()

    return Logs


def MNIST_Trainer(network, data, mypath, config):
    Xtrain, Ytrain, Xtest, Ytest, nclasses = data

    print("\nEvaluate network with no training:")
    TrainL0, TrainA0 = network.evaluate(Xtrain, Ytrain, batch_size=200, verbose=2)
    TestL0, TestA0 = network.evaluate(Xtest, Ytest, batch_size=200, verbose=2)
    neg, zero, pos = utils.getNZP(network)
    NZPMasks = [[neg, zero, pos]]
    denom = (neg + zero + pos)
    print("neg: {}, zero: {}, pos: {} - {:.7f}, {:.7f}, {:.7f}".format(neg, zero, pos, neg / denom, zero / denom, pos / denom))

    weights = utils.getkernels(network)
    file = open(mypath + "Weights0.pkl", "wb")
    pickle.dump(weights, file)
    file.close()

    file = open(mypath + "BestWeights.pkl", "wb")
    pickle.dump(weights, file)
    file.close()
    utils.plot_uniques(weights, mypath + "BestWeights.png")

    if config['standard_kernel'] == False:
        bits = utils.getbits(network)
        file = open(mypath + "Bits0.pkl", "wb")
        pickle.dump(bits, file)
        file.close()

    TrainLoss = np.asarray([TrainL0])
    TrainAccuracy = np.asarray([TrainA0])

    TestLoss = np.asarray([TestL0])
    TestAccuracy = np.asarray([TestA0])

    maxtrainacc = TrainA0
    maxtestacc = TestA0

    loss, metric = network.metrics_names

    expanded_lrates = training_schedule(config["lr_schedule"])
    batchsize = config["batchsize"]
    maxepochs = len(expanded_lrates)

    # custom train loop
    epoch = 0
    utils.plot_uniques(weights, mypath + "weights" + str(epoch) + ".png")
    while epoch < maxepochs:
        start_time = time.time()
        lr = expanded_lrates[epoch]
        KB.set_value(network.optimizer.lr, lr)

        if config["verbose"] == 1:
            weights = utils.getkernels(network)
            utils.plot_uniques(weights, mypath + "weights" + str(epoch) + ".png")

        fit_history = network.fit(Xtrain, Ytrain, batch_size=batchsize, epochs=1, verbose=0, shuffle=True, validation_data=(Xtest, Ytest))

        TrainLoss = np.append(TrainLoss, fit_history.history[loss])
        TestLoss = np.append(TestLoss, fit_history.history['val_loss'])

        TrainAccuracy = np.append(TrainAccuracy, fit_history.history[metric])
        TestAccuracy = np.append(TestAccuracy, fit_history.history['val_' + metric])

        if TestAccuracy[-1] > maxtestacc:
            predictions = network.predict(Xtest)
            np.save(mypath + "BestTestPredictions.npy", predictions)
            weights = utils.getkernels(network)
            file = open(mypath + "BestWeights.pkl", "wb")
            pickle.dump(weights, file)
            file.close()
            utils.plot_uniques(weights, mypath + "BestWeights.png")

        maxtrainacc = max(maxtrainacc, TrainAccuracy[-1])
        maxtestacc = max(maxtestacc, TestAccuracy[-1])

        print("\n")
        print("trainable bits  - {}".format(''.join(str(e) for e in config["trainableBits"])), len(config["trainableBits"]))
        print("learn rate      - {:.13f}".format(lr))
        print("trn             - loss: {:.13f} | acc {:.7f} | best: {:.7f} | avg 5: {:.7f}".format(TrainLoss[-1], TrainAccuracy[-1], maxtrainacc, np.mean(TrainAccuracy[-5:])))
        print("tst             - loss: {:.13f} | acc {:.7f} | best: {:.7f} | avg 5: {:.7f}".format(TestLoss[-1], TestAccuracy[-1], maxtestacc, np.mean(TestAccuracy[-5:])))
        neg, zero, pos = utils.getNZP(network)
        denom = (neg + zero + pos)
        NZPMasks.append([neg, zero, pos])
        print("nzp             - {} | {} | {}  -  {:.7f}, {:.7f}, {:.7f}".format(neg, zero, pos, neg / denom, zero / denom, pos / denom))
        print("epoch           - {}/{},  runtime: {:.3f} seconds".format(epoch + 1, maxepochs, time.time() - start_time))
        epoch += 1

    Logs = {"trainLoss": TrainLoss,
            "testLoss": TestLoss,
            "trainAccuracy": TrainAccuracy,
            "testAccuracy": TestAccuracy,
            "neg_zero_pos_masks": NZPMasks
            }

    np.savetxt(mypath + 'TrainAccuracy.txt', TrainAccuracy, delimiter=',')
    np.savetxt(mypath + 'TestAccuracy.txt', TestAccuracy, delimiter=',')

    predictions = network.predict(Xtest)
    np.save(mypath + "TestPredictions.npy", predictions)

    file = open(mypath + "TrainLogs.pkl", "wb")
    pickle.dump(Logs, file)
    file.close()

    weights = utils.getkernels(network)
    file = open(mypath + "Weights" + str(epoch) + ".pkl", "wb")
    pickle.dump(weights, file)
    file.close()
    utils.plot_uniques(weights, mypath + "weights" + str(epoch) + ".png")

    bits = utils.getbits(network)
    file = open(mypath + "Bits" + str(epoch) + ".pkl", "wb")
    pickle.dump(bits, file)
    file.close()

    return "Weights" + str(epoch) + ".pkl"


def Conv6(config):
    # some particular config stuff goes in here:
    # learning rates, batch size and epochs the same as ResNet, can/should be tuned

    config["batchsize"] = 64
    config["lr_schedule"] = [[150, 20, 30], [0.0006, 0.00006, 0.000006]]
    config["verbose"] = 0

    config["name"] = "Conv6" + "_w" + str(config["wbits"])
    mypath = utils.make_outputpath(config)
    if not os.path.exists(mypath):
        os.makedirs(mypath)

    runID = mypath.split('/')[-2]
    copy2(config["basedir"] + "/run_IDs/" + 'code_' + mainUUID + '.zip', mypath + "code_" + mainUUID + "_" + runID + ".zip")

    file = open(mypath + "Config.pkl", "wb")
    pickle.dump(config, file)
    file.close()

    data = cifar10()
    network = build_conv6(data[0].shape[1:], config)
    network.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
    network.summary()

    print("All files will be available in:", os.getcwd() + "/" + mypath)
    CIFAR_Trainer(network, data, mypath, config)
    print("All files available in:", os.getcwd() + "/" + mypath)
    KB.clear_session()
    return mypath


def ResNet(config):
    # some particular config stuff goes in here:
    # learning rates, batch size and epochs found by some older lr scan run, almost sure there are better ones
    # smarter learning rate schedules can/should be used

    config["batchsize"] = 64
    config["lr_schedule"] = [[150, 20, 30], [0.0006, 0.00006, 0.000006]]
    config["verbose"] = 0

    version, n = 1, 3
    config["name"] = "ResNet_V" + str(version) + "_n" + str(n) + "_w" + str(config["wbits"])
    mypath = utils.make_outputpath(config)
    if not os.path.exists(mypath):
        os.makedirs(mypath)

    runID = mypath.split('/')[-2]
    copy2(config["basedir"] + "/run_IDs/" + 'code_' + mainUUID + '.zip', mypath + "code_" + mainUUID + "_" + runID + ".zip")

    file = open(mypath + "Config.pkl", "wb")
    pickle.dump(config, file)
    file.close()

    data = cifar10()
    network = ResNetBuilder.MakeResNet(data[0].shape[1:], version, n, config)
    network.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
    network.summary()

    print("All files will be available in:", os.getcwd() + "/" + mypath)
    CIFAR_Trainer(network, data, mypath, config)
    print("All files available in:", os.getcwd() + "/" + mypath)
    KB.clear_session()

    return 0


def LeNet(config):
    config["name"] = "LeNet300_w" + str(config["wbits"])
    mypath = utils.make_outputpath(config)
    if not os.path.exists(mypath):
        os.makedirs(mypath)
    runID = mypath.split('/')[-2]

    # make another copy of the code files in this particular folder
    copy2(config["basedir"] + "/run_IDs/" + 'code_' + mainUUID + '.zip', mypath + "code_" + mainUUID + "_" + runID + ".zip")

    file = open(mypath + "Config.pkl", "wb")
    pickle.dump(config, file)
    file.close()

    network = build_LeNet300(config)
    network.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
    network.summary()

    print("Start run -> All files will be available in:", os.getcwd() + "/" + mypath)
    MNIST_Trainer(network, mnist(), mypath, config)
    print("End run -> All files available in:", os.getcwd() + "/" + mypath)
    KB.clear_session()
    return


def main():
    prefix = "BitDepth/"

    # Number of bits on which weights are encoded, the larger it is, the slower it trains.
    nbits = 6

    # First bit is the least significant bit, last bit is the sign bit.
    trainableBits = [0, 0, 0, 1, 1, 1]

    leafdir = "bits_"
    leafdir += str(len(trainableBits))
    leafdir += "/"

    # For normal training this is empty, if we want to load pre-trained weights, it should specify
    # the path to the pre-treind weights and the way they are encoded: (f)loats, (i)ntegers or raw (b)its
    # pretrained_bittensor = [["path/to/file", "f"]]  (will updated in a later version of this code)
    pretrained_bittensor = []

    # Fhis specifies the locations of where to place the pretrained bits.
    # (will updated in a later version of this code)
    pretrained_bitplacement = []

    # This configures the training procedure and the dense/conv layers.
    # Anything may be overwritten in the training function.
    config = {
        "basedir": "Outputs/" + prefix + leafdir,  # where to save the results
        "verbose": 0,  # if 1 then it saves the histogram of the weights for all epochs
        "initializer": "he",  # used for initializing weights, see localLayers.py
        "activation": 'relu',  # some obviously named parameter
        "standard_kernel": False,  # is this is true then it trains in the standard way, using default float32 weights
        "lr_schedule": [[40, 40, 20], [0.0009, 0.00009, 0.000009]],  # LeNet may actually converge in <15 epochs, tune for better results
        "batchsize": 25,  # some other obviously named parameter
        "wbits": len(trainableBits),  # bit-depth used for weights
        "trainableBits": trainableBits,  # specify which bits are trainable, kind related to the previous
        "pretrained_bittensor": pretrained_bittensor,  # for custom initialized weights; can be anything representable in bits
        "pretrained_bitplacement": pretrained_bitplacement,  # for custom initialized weights, specifies where in the bit-string to place the above bits
        "inference_sequence": [0, nbits - 1]  # this allows us to choose which bits participate in the calculation
    }

    # This makes a copy of the python files involved in a run, just to save
    # all parameters in case too much experimentation makes the results break down.
    utils.copyfilesto(config["basedir"] + "/run_IDs/", mainUUID)

    # Run any of the functions below:

    LeNet(config)
    # ResNet(config)
    # Conv6(config)

    return


if __name__ == '__main__':
    main()
