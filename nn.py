# Bounnoy Phanthavong (ID: 973081923)
# Homework 2
#
# This is a machine learning program that models a neural network.
# Here, we implement a two-layer neural network with one hidden-layer
# to perform handwritten digit recognition.
# This program was built in Python 3.

from pathlib import Path
from math import e
import numpy as np
import csv
import pickle

class NeuralNetwork:
    def __init__(self, train, test):
        self.trainData = train
        self.testData = test

    # The train function takes in the ETA (learning rate), iterations, hidden units,
    # and output units, then outputs results.
    def train(self, eta, iterations, hunits, output):
        accuracy = np.zeros(iterations)         # Training accuracy.
        accuracyTest = np.zeros(iterations)     # Test accuracy.

        # Delete later.
        print("\nTraining Accuracy: ", accuracy)
        print("Array Size: ", len(accuracy))
        print("\nTest Accuracy: ", accuracyTest)
        print("Array Size: ", len(accuracyTest))

        # PREPARE INPUT VECTORS
        rowsTrain = len(self.trainData)         # Rows in the training data.
        rowsTest = len(self.testData)           # Rows in the test data.

        # Shuffle training/testing data.
        #np.random.shuffle(self.trainData)
        #np.random.shuffle(self.testData)

        tTrain = self.trainData[:,0]        # Set training target to first column of training data.
        tTrain = np.vstack(tTrain)          # Convert it to a vertical array.
        xTrain = self.trainData[:,1:]       # Set inputs as everything after first column.
        #xTrain = xTrain/255                 # Divide all cells to keep calculation small. (0-1)

        # Delete later.
        print("\nTraining Targets:\n", tTrain)
        print("Rows/Columns: ", len(tTrain), "/", len(tTrain[0]))
        print("\nTraining Input:\n", xTrain)
        print("Rows/Columns: ", len(xTrain), "/", len(xTrain[0]))

        # Do the same as above for testing set.
        tTest = self.testData[:,0]
        tTest = np.vstack(tTest)
        xTest = self.testData[:,1:]
        #xTest = xTest/255

        # Delete later.
        print("\nTest Targets:\n", tTest)
        print("Rows/Columns: ", len(tTest), "/", len(tTest[0]))
        print("\nTest Input:\n", xTest)
        print("Rows/Columns: ", len(xTest), "/", len(xTest[0]))

        # Replace first column with the bias.
        xTrain = np.concatenate( (np.ones((rowsTrain, 1)), xTrain), axis=1 )
        xTest = np.concatenate( (np.ones((rowsTest, 1)), xTest), axis=1 )

        # Delete later.
        print("\nTraining Input with Bias:\n", xTrain)
        print("Rows/Columns: ", len(xTrain), "/", len(xTrain[0]))
        print("\nTest Input with Bias:\n", xTest)
        print("Rows/Columns: ", len(xTest), "/", len(xTest[0]))

        # PREPARE HIDDEN LAYER
        hTrain = np.zeros((rowsTrain, hunits))
        hTest = np.zeros((rowsTest, hunits))

        # Delete later.
        print("\nTraining Hidden Layer:\n", hTrain)
        print("Rows/Columns: ", len(hTrain), "/", len(hTrain[0]))
        print("\nTest Hidden Layer:\n", hTest)
        print("Rows/Columns: ", len(hTest), "/", len(hTest[0]))

        # Replace first column with the bias.
        hTrain = np.concatenate( (np.ones((rowsTrain, 1)), hTrain), axis=1 )
        hTest = np.concatenate( (np.ones((rowsTest, 1)), hTest), axis=1 )

        # Delete later.
        print("\nTraining Hidden Layer with Bias:\n", hTrain)
        print("Rows/Columns: ", len(hTrain), "/", len(hTrain[0]))
        print("\nTest Hidden Layer with Bias:\n", hTest)
        print("Rows/Columns: ", len(hTest), "/", len(hTest[0]))

        # PREPARE OUTPUT LAYER
        oTrain = np.zeros((rowsTrain, output))
        oTest = np.zeros((rowsTest, output))

        # Delete later.
        print("\nTraining Output Layer:\n", oTrain)
        print("Rows/Columns: ", len(oTrain), "/", len(oTrain[0]))
        print("\nTest Output Layer:\n", oTest)
        print("Rows/Columns: ", len(oTest), "/", len(oTest[0]))

        # PREPARE WEIGHT VECTORS

        # Create a 2D array with the same rows as the output and cols of training data.
        # Weights are populated randomly between -0.5 to 0.5 for each cell.
        #weights = np.random.rand(len(train[0]), 2)*.1-.05
        #print(weights)
        hWeights = np.array([[.15, .20, .25, .30],
                             [.15, .20, .25, .30]])

        # Delete later.
        print("\nInput > Hidden Weight Vector:\n", hWeights)
        print("Rows/Columns: ", len(hWeights), "/", len(hWeights[0]))

        oWeights = np.array([[.40, .45, .50, .55],
                             [.40, .45, .50, .55]])

        # Delete later.
        print("\nHidden > Output Weight Vector:\n", oWeights)
        print("Rows/Columns: ", len(oWeights), "/", len(oWeights[0]))

        # FORWARD PROPOGATION: Sigmoid Activation
        b = np.array([.35, .60])
        h = np.array([])
        for j in range(hunits):
            offset = j * 2
            z = np.dot(xTrain[0], np.concatenate((b[0], hWeights[0][offset:offset+2]), axis=None))
            print("\nz =", z)

            h = np.append(h, [1/(1 + e**(-z))])
            print("h[", j, "] =", h[j])

        o = np.array([])
        h = np.concatenate( ([1], h), axis=None )
        for k in range(output):
            offset = k * 2
            z = np.dot(h, np.concatenate((b[1], oWeights[0][offset:offset+2]), axis=None))
            print("\nz =", z)

            o = np.append(o, [1/(1 + e**(-z))])
            print("o[", k, "] =", o[k])

if __name__ == '__main__':

    pklTrain = Path("mnist_train.pkl")
    pklTest = Path("mnist_test.pkl")
    fileTrain = Path("mnist_train.csv")
    fileTest = Path("mnist_test.csv")

    if not fileTrain.exists():
        sys.exit("mnist_train.csv not found")

    if not fileTest.exists():
        sys.exit("mnist_test.csv not found")

    if not pklTrain.exists():
        f = np.genfromtxt("mnist_train.csv", delimiter=",")
        csv = open("mnist_train.pkl", 'wb')
        pickle.dump(f, csv)
        csv.close()

    if not pklTest.exists():
        f = np.genfromtxt("mnist_test.csv", delimiter=",")
        csv = open("mnist_test.pkl", 'wb')
        pickle.dump(f, csv)
        csv.close()

    file = open("mnist_train.pkl", "rb")
    train = pickle.load(file)
    file.close()

    file = open("mnist_test.pkl", "rb")
    test = pickle.load(file)
    file.close()

    # Delete later.
    print("\nRows in training data: ", len(train))
    print("Cols in training data: ", len(train[0]))

    # Delete later.
    print("\nRows in test data: ", len(test))
    print("Cols in test data: ", len(test[0]))

    output = 2

    nn = NeuralNetwork(train, test)

    nn.train(0.1, 1, 2, output)
