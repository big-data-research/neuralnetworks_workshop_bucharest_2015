__author__ = 'alexs'

import cPickle
import random

import theano.tensor as T
import theano
import numpy as np


def getReferenceLabels():
    referenceLabels = dict()
    for i in range(0, 10):
        reference_out = [0.1 for x in range(0, 10)]
        reference_out[i] = 0.88
        referenceLabels[i] = reference_out
    return referenceLabels


def sigmoid(x):
    return 1.0 / (1 + T.exp(-1.0 * x))


def compare(result_label, given_label, reference_labels):
    givenKey = 0
    resultedKey = 0
    refGivenScore = 1000
    refResultedScore = 1000

    for key in reference_labels.keys():
        score1 = np.sum(np.abs(np.array(given_label) - np.array(reference_labels[key])))
        score2 = np.sum(np.abs(result_label - np.array(reference_labels[key])))
        if score1 < refGivenScore:
            refGivenScore = score1
            givenKey = key
        if score2 < refResultedScore:
            refResultedScore = score2
            resultedKey = key

    if resultedKey == givenKey:
        return True
    return False


def makeW(rows, columns, start=-2, end=2):
    w = np.random.uniform(start, end, (rows, columns))
    return w


def updates_weights_function(weights, memories, cost_function, learning_rate=0.2, momentum_learning_rate=0.05):
    gradients = T.grad(cost_function, weights)  # keep in mind len(gradients) == len(weights)

    update_lists = []
    for i in range(0, len(weights)):
        weight = weights[i]
        gradient = gradients[i]
        memory = memories[i]
        change = learning_rate * gradient + momentum_learning_rate * memory
        new_val = weight - change
        update_lists.append((weight, new_val))
        update_lists.append((memory, change))
    return update_lists


class NN():
    def __init__(self):
        self.layers = []
        self.weights = []
        self.weights_memory = []
        self.cost = None
        self.train = None
        self.updates = None
        self.activate = None
        self.activatwe = None
        self.output = None


    def build(self, givenWeights=None):
        # first: init or build the in-between weight matrixes
        for i in range(0, len(self.layers) - 1):
            n = self.layers[i].size
            m = self.layers[i + 1].size
            if givenWeights:
                w_values = givenWeights[i]
            else:
                w_values = makeW(n, m)
            w_memory_values = np.zeros((n, m))
            w = theano.shared(value=w_values, name="w_" + str(i) + "_" + str(i + 1))
            w_memory = theano.shared(value=w_memory_values, name="w_memory_" + str(i) + "_" + str(i + 1))
            self.weights.append(w)
            self.weights_memory.append(w_memory)

        # now build the model
        inputVector = T.matrix("inputVector")
        labels = T.matrix("labels")

        out = None
        net = None

        workingV = inputVector

        l2 = 0.0
        l1 = 0.0

        for i in range(0, len(self.weights)):
            w = self.weights[i]
            l2 += T.sum(w * w)
            l1 += T.sum(T.abs_(w))
            out = T.dot(workingV, w)
            net = sigmoid(out)
            workingV = net

        self.cost = T.sum(T.pow(labels - net, 2))  # + 0.005 * l2 # + 0.005 * l1
        self.output = net

        self.updates = updates_weights_function(self.weights, self.weights_memory, self.cost)
        self.train = theano.function([inputVector, labels], outputs=self.cost, updates=self.updates)
        self.activate = theano.function([inputVector, labels], outputs=self.cost)
        self.activatwe = theano.function([inputVector], outputs=self.output)


    def addLayer(self, layer):
        self.layers.append(layer)


    def trainData(self, train_set_input, train_set_labels,
                  valid_set_input, valid_set_labels,
                  test_set_input, test_set_labels,
                  nrOfEpochs=10000, batch_size=1000):

        reference_labels = getReferenceLabels()
        for ep in range(0, nrOfEpochs):
            overallError = 0.0
            for j in range(0, len(train_set_input), batch_size):
                endInterval = j + batch_size
                if j + batch_size > len(train_set_input):
                    endInterval = len(train_set_input) - 1
                i = train_set_input[j:endInterval]
                r = train_set_labels[j:endInterval]
                self.train(i, r)

            for j in range(0, len(train_set_input), batch_size):
                endInterval = j + batch_size
                if j + batch_size > len(train_set_input):
                    endInterval = len(train_set_input) - 1
                i = train_set_input[j:endInterval]
                r = train_set_labels[j:endInterval]
                overallError += self.activate(i, r)

            posItems = 0.0
            failedItems = 0.0
            for valid_in, given_label in zip(valid_set_input, valid_set_labels):
                result_label = self.activatwe([valid_in])
                ok = compare(result_label, given_label, reference_labels)
                if ok:
                    posItems += 1.0
                else:
                    failedItems += 1.0

            precision = posItems / (posItems + failedItems)

            print(
                "[{epoch}] error: {error} precision: {precision}".format(epoch=ep, error=overallError,
                                                                         precision=precision))

        # running tests
        if test_set_input and test_set_labels:
            print("=================== TESTS ==================")
            posItems = 0.0
            failedItems = 0.0
            for valid_in, given_label in zip(test_set_input, test_set_labels):
                result_label = self.activatwe([valid_in])
                ok = compare(result_label, given_label, reference_labels)
                if ok:
                    posItems += 1.0
                else:
                    failedItems += 1.0

            precision = posItems / (posItems + failedItems)
            print("Accuracy on {nrOfTests} tests is {precision}".format(nrOfTests=str(len(test_set_input)),
                                                                        precision=str(precision)))
            print("============================================")


class Layer():
    def __init__(self, size):
        self.size = size


class SigmoidLayer(Layer):
    def __init__(self, size):
        self.size = size


class StandardOutputWithSigmoid(Layer):
    def __init__(self, size):
        self.size = size


def retrieveTrainValidationTest():
    f = open("mnist.pkl")
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return train_set, valid_set, test_set


def processData(nnset, sampleSize=None):
    train_in = nnset[0]
    train_label = nnset[1]
    d = {}
    for index in range(0, len(train_label)):
        label = train_label[index]
        d.setdefault(label, []).append(train_in[index])

    if sampleSize:
        d_sample = []
        for key in d.keys():
            for train_in in d[key][0:sampleSize]:
                d_sample.append((key, train_in))
    else:
        d_sample = []
        for key in d.keys():
            for train_in in d[key]:
                d_sample.append((key, train_in))
    random.shuffle(d_sample)

    results_in = []
    results_label_out = []
    for i in range(0, len(d_sample)):
        label = d_sample[i][0]
        train_in = d_sample[i][1]
        # now create the arrays
        label_out = [0.1 for x in range(0, 10)]
        label_out[label] = 0.88
        results_in.append(np.array(train_in, dtype="float32"))
        results_label_out.append(np.array(label_out, dtype="float32"))
    return results_in, results_label_out


def main():
    nn = NN()
    nn.addLayer(SigmoidLayer(784))
    nn.addLayer(SigmoidLayer(100))
    nn.addLayer(StandardOutputWithSigmoid(10))
    nn.build()

    train_set, valid_set, test_set = retrieveTrainValidationTest()
    # TRAINING_SAMPLE_SIZE = 100;    VALIDATION_SAMPLE_SIZE = 10;    TEST_SAMPLE_SIZE = 10
    TRAINING_SAMPLE_SIZE = VALIDATION_SAMPLE_SIZE = TEST_SAMPLE_SIZE = None

    train_set_input, train_set_labels = processData(train_set, TRAINING_SAMPLE_SIZE)
    valid_set_input, valid_set_labels = processData(valid_set, VALIDATION_SAMPLE_SIZE)
    test_set_input, test_set_labels = processData(test_set, TEST_SAMPLE_SIZE)
    nn.trainData(train_set_input, train_set_labels,
                 valid_set_input, valid_set_labels,
                 test_set_input, test_set_labels,
                 nrOfEpochs=10, batch_size=100)


if __name__ == '__main__':
    main()