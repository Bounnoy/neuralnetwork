# Bounnoy Phanthavong (ID: 973081923)
# Homework 2
#
# This is a machine learning program that models a neural network.
# Here, we implement a two-layer neural network with one hidden-layer
# to perform handwritten digit recognition.
# This program was built in Python 3.

from pathlib import Path
import numpy as np
import csv
import pickle

class NeuralNetwork:
    def __init__(self, train, test, output):
        self.trainData = train
        self.testData = test
        self.output = output

    # The train function takes in the ETA (learning rate), iterations, and hidden units,
    # then outputs results.
    def train(self, eta, iterations, hunits):
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
        np.random.shuffle(self.trainData)
        np.random.shuffle(self.testData)

        tTrain = self.trainData[:,0]        # Set training target to first column of training data.
        tTrain = np.vstack(tTrain)          # Convert it to a vertical array.
        xTrain = self.trainData[:,1:]       # Set inputs as everything after first column.
        xTrain = xTrain/255                 # Divide all cells to keep calculation small. (0-1)

        # Delete later.
        print("\nTraining Targets:\n", tTrain)
        print("Rows/Columns: ", len(tTrain), "/", len(tTrain[0]))
        print("\nTraining Input:\n", xTrain)
        print("Rows/Columns: ", len(xTrain), "/", len(xTrain[0]))

        # Do the same as above for testing set.
        tTest = self.testData[:,0]
        tTest = np.vstack(tTest)
        xTest = self.testData[:,1:]
        xTest = xTest/255

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

        # PREPARE WEIGHT VECTORS

        # Create a 2D array with the same rows as the output and cols of training data.
        # Weights are populated randomly between -0.5 to 0.5 for each cell.
        # weights = np.random.rand(len(train[0]), self.output)*.1-.05

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

    output = 10

    nn = NeuralNetwork(train, test, output)
    nn.train(0.1, 50, 20)
