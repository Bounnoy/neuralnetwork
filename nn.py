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
        hWeights = np.full((rowsTrain, hunits, len(xTrain[0])), 0.1, dtype=float)

        # Delete later.
        print("\nInput > Hidden Weight Vector:\n", hWeights)
        print("Rows/Columns: ", len(hWeights), "/", len(hWeights[0]))

        oWeights = np.full((rowsTrain, output, hunits + 1), 0.1, dtype=float)

        # Delete later.
        print("\nHidden > Output Weight Vector:\n", oWeights)
        print("Rows/Columns: ", len(oWeights), "/", len(oWeights[0]))

        # FORWARD PROPOGATION: Sigmoid Activation
        for i in range(rowsTrain):
            #b = np.array([.1, .1, .1])
            #h = np.array([1])
            t = np.full(output, 0.1, dtype=float)
            t[ int(tTrain[i]) ] = 0.9

            # for j in range(hunits):
            #     #offset = j * hunits
            #     z = np.dot(hWeights[i][j], np.vstack(xTrain[i]))#, hWeights[i])#np.concatenate((b[0:hunits], hWeights[i][offset:offset + hunits]), axis=None))
            #     print("\nz =", z)
            #
            #     h = np.append(h, [1/(1 + e**(-z))])
            #     print("h[", j, "] =", h[j])

            z = np.dot(hWeights[i], np.vstack(xTrain[i]))
            print("\nz =", z)
            h = 1/(1 + e**(-z))# np.array([1, [1/(1 + e**(-z))])
            h = np.concatenate(([[1]], h), axis=0)
            print("h =", h)

            #o = np.array([])
            #errorO = np.array([])
            #dk = np.array([])
            #dj = np.array([])
            #h = np.concatenate( ([1], h), axis=None )

            print("\nh =", h)
            print("cols =", len(h))
            # for k in range(output):
            #     #offset = k * output + 1
            #     z = np.dot(oWeights[i], np.vstack(h))#, oWeights[i])#np.concatenate((b[hunits:], oWeights[i][offset:offset + output + 1]), axis=None))
            #     print("\nz =", z)
            #
            #     o = np.append(o, [1/(1 + e**(-z))])
            #     print("o[", k, "] =", o[k])
            #
            #     dk = np.append(dk, [o[k]*(1-o[k])*(t[k]-o[k])])
            #     print("dk[", k, "] = ", dk[k])
            z = np.dot(oWeights[i], np.vstack(h))
            print("\nz =", z)
            o = 1/(1 + e**(-z))
            print("o =", o)
            dk = o*(1-o)*(t-o)
            print("dk =", dk)

            for j in range(hunits + 1)[1:]:
                dj = h*(1-h)*(np.dot(oWeights[i,...,j], dk))#np.append(dj, [h[j]*(1-h[j])*(np.dot(oWeights[i][offset:offset+output], dk))])
                print("dj[", j, "] = ", dj[j])
            #dj = h[1:]*(1-h[1:])*(np.dot(oWeights[i][:][1:], dk))
            #print("dj =", dj)

            #oWeights[i] += eta * np.dot(dk, h)


            #print(oWeights[i])
            # Calculate total error.

            #errorO = np.append(errorO, [0.5*(t[k] - o[k])**2])


            print("\nt[", int(tTrain[i]), "] =", t[ int(tTrain[i]) ])
            #prediction = np.argmax(o)
            #print(prediction)
            #print("output error[", k, "] =", errorO[k])



            #print("total output error =", np.sum(errorO))

        # BACKWARD PROPOGATION


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

    output = 1

    nn = NeuralNetwork(train, test)

    nn.train(0.1, 1, 2, output)
