__author__ = 'alexs'

import theano.tensor as T
import theano
import numpy as np
import cPickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import random
import json


def getReferenceLabels():
    referenceLabels = dict()
    for i in range(0, 9):
        reference_out = [0.0 for x in range(0, 9)]
        reference_out[i] = 0.99
        referenceLabels[i] = reference_out
    return referenceLabels


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


def updates_weights_function(weights, memories, cost_function, learning_rate=0.01, momentum_learning_rate=0.005):
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
            net = T.maximum(0, out)
            workingV = net

        self.cost = T.mean(T.pow(labels - out, 2)) + 0.005 * l2 + 0.005 * l1
        self.output = net

        self.updates = updates_weights_function(self.weights, self.weights_memory, self.cost)
        self.train = theano.function([inputVector, labels], outputs=self.cost, updates=self.updates)
        self.activate = theano.function([inputVector, labels], outputs=self.cost)
        self.activatwe = theano.function([inputVector], outputs=self.output)


    def addLayer(self, layer):
        self.layers.append(layer)

    def snapshotWeigths(self, experimentId):
        with open(str(experimentId) + ".dat", "w") as f:
            for w in self.weights:
                numeric_value = w.get_value().tolist()
                f.write(json.dumps(numeric_value) + "\n")

    def resume(self, experimentId="default"):
        ww = []
        with open(str(experimentId) + ".dat", "r") as f:
            for line in f.readlines():
                w = np.array(json.loads(line))
                ww.append(w)
        self.build(ww)


    def trainData(self, train_set_input, train_set_labels,
                  valid_set_input, valid_set_labels,
                  test_set_input, test_set_labels,
                  nrOfEpochs=10000, batch_size=1000, experimentId="default"):

        reference_labels = getReferenceLabels()
        for ep in range(0, nrOfEpochs):
            # random.shuffle(train_data)
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
        self.snapshotWeigths(experimentId)
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


class InverseOutputLayerWithSigmoid(Layer):
    def __init__(self, size):
        self.size = size


def transformInput(inputList):
    res = []
    for input in inputList:
        res.append(np.array(input, dtype="float32"))
    return res


def transformOutput(outputList, size):
    res = []
    for out in outputList:
        reference_out = [0.1 for x in range(0, size)]
        reference_out[out] = 0.88
        res.append(np.array(reference_out, dtype="float32"))
    return res


def retrieve_training_set():
    all_collections = []
    df = pd.read_csv("/Users/alexs/work_phd/otto_group_challenge/train.csv")
    X = df.values.copy()
    np.random.shuffle(X)
    X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
    print labels
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels).astype(np.int32)
    all_labels = []
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    for encoded_label in encoded_labels:
        l = [0.0 for x in range(0, 9)]
        l[encoded_label] = 0.99
        all_labels.append(l)
    return [Z, all_labels]


def retrieveTrainValidationSet(train_set, percentage=20.0):
    train = train_set[0]
    label = train_set[1]
    all = []
    for i in range(0, len(train)):
        all.append((train[i], label[i]))
    random.shuffle(all)
    offset = int(len(train) * (percentage / 100.0))
    validation_final = []
    validation_input = []
    validation_label = []
    for i in range(0, offset):
        (vi, vl) = all.pop(0)
        validation_input.append(vi)
        validation_label.append(vl)
    validation_final.append(validation_input)
    validation_final.append(validation_label)

    training_final = []
    training_in = []
    training_label = []
    for (ti, tl) in all:
        training_in.append(ti)
        training_label.append(tl)
    training_final.append(training_in)
    training_final.append(training_label)

    return training_final, validation_final


def retrieve_test_set():
    df = pd.read_csv("/Users/alexs/work_phd/otto_group_challenge/test.csv")
    X = df.values.copy()
    np.random.shuffle(X)
    X = X[:, 1:].astype(np.float32)

    scaler = StandardScaler()
    Z = scaler.fit_transform(X)

    return Z


def getClosest(out, reference):
    refGivenScore = 1000
    givenKey = 0

    for key in reference.keys():
        score1 = np.sum(np.abs(np.array(out) - np.array(reference[key])))
        if score1 < refGivenScore:
            refGivenScore = score1
            givenKey = key
    # p = [0 for i in range(0,9)]
    # p[key]=1
    cleaned_p = []
    for p in reference[givenKey]:
        if p < 0.4:
            cleaned_p.append(0)
        elif p > 0.8:
            cleaned_p.append(0.95)
        else:
            cleaned_p.append(p)

    return [str(p) for p in cleaned_p]




def main():

    nn = NN()
    nn.addLayer(SigmoidLayer(93))
    nn.addLayer(SigmoidLayer(30))
    nn.addLayer(InverseOutputLayerWithSigmoid(9))
    nn.build()
    # nn.resume()

    original_data = retrieve_training_set()
    test_data = retrieve_test_set()
    batch_size = 2000
    for i in range(0, 100):
        print("BIG ITERATION: " + str(i))
        training_set, validation_set = retrieveTrainValidationSet(original_data, percentage=30)
        train_set_input, train_set_labels = training_set[0], training_set[1]
        valid_set_input, valid_set_labels = validation_set[0], validation_set[1]
        nn.trainData(train_set_input, train_set_labels,
                     valid_set_input, valid_set_labels,
                     None, None,
                     nrOfEpochs=10, batch_size=batch_size)
        batch_size = batch_size - 50
        if (batch_size < 100):
            batch_size = 100

    print("RUNNING THE TESTS")
    referenceLabels = getReferenceLabels()

    with open("submission.dat", "w") as w:
        w.write("id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n")
        counter = 1
        for test in test_data:
            resultedLabel = nn.activatwe([test])
            out = getClosest(resultedLabel, referenceLabels)
            #w.write(str(counter) + "," + ",".join(out) + "\n")
            a = [str(p) for p in resultedLabel[0]]
            w.write(str(counter) + "," + ",".join(a) + "\n")
            counter += 1


if __name__ == '__main__':
    main()