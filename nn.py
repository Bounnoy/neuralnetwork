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
import time

class NeuralNetwork:
    def __init__(self, train, test):
        self.trainData = train
        self.testData = test

    # The train function takes in the ETA (learning rate), iterations, hidden units,
    # and output units, then outputs results.
    def train(self, eta, iterations, hunits, output):
        accuracy = np.zeros(iterations)         # Training accuracy.
        accuracyTest = np.zeros(iterations)     # Test accuracy.

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

        # Do the same as above for testing set.
        tTest = self.testData[:,0]
        tTest = np.vstack(tTest)
        xTest = self.testData[:,1:]
        #xTest = xTest/255

        # Replace first column with the bias.
        xTrain = np.concatenate( (np.ones((rowsTrain, 1)), xTrain), axis=1 )
        xTest = np.concatenate( (np.ones((rowsTest, 1)), xTest), axis=1 )

        # PREPARE HIDDEN LAYER
        hTrain = np.zeros((rowsTrain, hunits))
        hTest = np.zeros((rowsTest, hunits))

        # Replace first column with the bias.
        hTrain = np.concatenate( (np.ones((rowsTrain, 1)), hTrain), axis=1 )
        hTest = np.concatenate( (np.ones((rowsTest, 1)), hTest), axis=1 )

        # PREPARE OUTPUT LAYER
        oTrain = np.zeros((rowsTrain, output))
        oTest = np.zeros((rowsTest, output))

        # PREPARE WEIGHT VECTORS

        # Create a 2D array with the same rows as the output and cols of training data.
        # Weights are populated randomly between -0.5 to 0.5 for each cell.
        hWeights = np.random.rand(rowsTrain, hunits, len(xTrain[0]))*.1-.05
        oWeights = np.random.rand(rowsTrain, output, hunits + 1)*.1-.05

        # FORWARD PROPOGATION: Sigmoid Activation

        start = time.time()
        for n in range(iterations):
            correct = 0
            error = np.array([])
            for i in range(rowsTrain):
                t = np.full(output, 0.1, dtype=float)
                t[ int(tTrain[i]) ] = 0.9

                z = np.dot(hWeights[i], np.vstack(xTrain[i]))
                h = 1/(1 + e**(-z))
                h = np.concatenate(([[1]], h), axis=0)

                z = np.dot(oWeights[i], np.vstack(h))
                o = 1/(1 + e**(-z))
                dk = o*(1-o)*(t-o)

                for j in range(hunits + 1):
                    dj = h*(1-h)*(np.dot(oWeights[i,...,j], dk))
                    oWeights[i,...,j] += eta * np.dot(dk, h[j])

                for x in range(len(xTrain[i])):
                    hWeights[i,...,x] += eta * np.dot(dj[i], xTrain[i][x])

                # Calculate total error.
                error = np.append(error, [0.5*(t - o)**2])

                if t[np.argmax(o)] == 0.9:
                    correct += 1

            mse = round(np.sum(error) / rowsTrain, 6)
            accuracy[n] = ( float(correct) / float(rowsTrain) ) * 100
            end = time.time()
            elapsed = round((end - start)/60, 2)
            print("Epoch", n, ": Training Accuracy =", accuracy[n], "%, Error =", mse, "%, Elapsed Time =", elapsed, "min")


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

    nn.train(0.1, 50, 20, output)
