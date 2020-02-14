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
        np.random.shuffle(self.trainData)
        np.random.shuffle(self.testData)

        tTrain = self.trainData[:,0]        # Set training target to first column of training data.
        tTrain = np.vstack(tTrain)          # Convert it to a vertical array.
        xTrain = self.trainData[:,1:]       # Set inputs as everything after first column.
        xTrain = xTrain/255                 # Divide all cells to keep calculation small. (0-1)

        # Do the same as above for testing set.
        tTest = self.testData[:,0]
        tTest = np.vstack(tTest)
        xTest = self.testData[:,1:]
        xTest = xTest/255

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

        print("Learning Rate = ", eta)
        with open('results.csv', 'a') as csvFile:
            w = csv.writer(csvFile)
            w.writerow(["Learning Rate"] + [eta])
            w.writerow(["Epoch"] + ["Training Accuracy"] + ["Test Accuracy"])

        start = time.time()
        for n in range(iterations):
            correct = 0
            correctTest = 0
            error = np.array([])
            errorTest = np.array([])

            for i in range(rowsTrain):
                t = np.full(output, 0.1, dtype=float)
                t[ int(tTrain[i]) ] = 0.9

                z = hWeights[i].dot(np.vstack(xTrain[i]))
                h = 1/(1 + e**(-z))
                h = np.concatenate(([[1]], h), axis=0)

                z = oWeights[i].dot(np.vstack(h))
                o = 1/(1 + e**(-z))
                dk = o*(1-o)*(np.vstack(t)-o)
                dj = h*(1-h)*(np.sum(oWeights[i] * dk))

                oWeights[i] += eta * np.outer(dk, h)
                hWeights[i] += eta * np.outer(dj[1:], xTrain[i])
                
                # Calculate total error.
                error = np.append(error, [0.5*(t - o)**2])

                pr = np.argmax(o)
                if pr == tTrain[i]:
                    correct += 1

            for a in range(rowsTest):
                t = np.full(output, 0.1, dtype=float)
                t[ int(tTest[a]) ] = 0.9

                z = hWeights[a].dot(np.vstack(xTest[a]))
                h = 1/(1 + e**(-z))
                h = np.concatenate(([[1]], h), axis=0)

                z = oWeights[a].dot(np.vstack(h))
                o = 1/(1 + e**(-z))

                # Calculate total error.
                errorTest = np.append(errorTest, [0.5*(t - o)**2])

                pr = np.argmax(o)
                if pr == tTest[a]:
                    correctTest += 1

            mse = round(np.sum(error) / rowsTrain, 6)
            mseTest = round(np.sum(errorTest) / rowsTest, 6)

            accuracy[n] = ( float(correct) / float(rowsTrain) ) * 100
            accuracyTest[n] = ( float(correctTest) / float(rowsTest) ) * 100
            end = time.time()
            elapsed = round((end - start)/60, 2)
            print("Epoch", n, ": Training Acc. =", accuracy[n], "%, Error =", mse,
                            "%, Test Acc. =", accuracyTest[n], "%, Error =", mseTest, "%, Elapsed Time =", elapsed, "min")

            with open('results.csv', 'a') as csvFile:
                w = csv.writer(csvFile)
                w.writerow([n] + [accuracy[n]] + [accuracyTest[n]])

            if accuracy[int(n-1)] > (accuracy[n] + 1):
                break
        return hWeights, oWeights

    # Build the confusion matrix.
    def confusion(self, hunits, output, hWeights, oWeights):
        if (len(self.testData[0]) != len(self.trainData[0])):
            print("Error: Training and test data structure does not match.")
            return

        rowsTest = len(self.testData)
        np.random.shuffle(self.testData)    # Shuffle test data.
        tTest = self.testData[:,0]              # Set test target to first column of test data.
        tTest = np.vstack(tTest)                    # Convert it to a vertical array.
        xTest = self.testData[:,1:]             # Set inputs as everything after the first column.
        xTest = xTest/255                           # Divide all cells to keep calculation small. (0-1)

        # Replace first column with the bias.
        xTest = np.concatenate( (np.ones((rowsTest, 1)), xTest), axis=1 )

        # PREPARE HIDDEN LAYER
        hTest = np.zeros((rowsTest, hunits))

        # Replace first column with the bias.
        hTest = np.concatenate( (np.ones((rowsTest, 1)), hTest), axis=1 )

        # PREPARE OUTPUT LAYER
        oTest = np.zeros((rowsTest, output))

        matrix = np.zeros((output, output)) # Build our matrix.
        testAccuracy = 0

        for a in range(rowsTest):
            t = np.full(output, 0.1, dtype=float)
            t[ int(tTest[a]) ] = 0.9

            z = hWeights[a].dot(np.vstack(xTest[a]))
            h = 1/(1 + e**(-z))
            h = np.concatenate(([[1]], h), axis=0)

            z = oWeights[a].dot(np.vstack(h))
            o = 1/(1 + e**(-z))

            pr = np.argmax(o)
            if pr == tTest[a]:
                correctTest += 1

            # Plot our data in the table if correct prediction.
            matrix[int(prediction)][int(tTest[a])] += 1

        # Calculate test accuracy.
        accuracy = int( (float(correctTest)/float(rowsTest)) * 100)

        print("Final Accuracy = ", accuracy, "%")

        np.set_printoptions(suppress = True)
        print("\nConfusion Matrix")
        print(matrix, "\n")

        with open('results.csv', 'a') as csvFile:
            w = csv.writer(csvFile)
            w.writerow([])
            w.writerow(["Confusion Matrix"])
            for j in range(output):
                w.writerow(matrix[j,:])
            w.writerow(["Final Accuracy"] + [accuracy])
            w.writerow([])

        return

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

    output = 10

    # EXPERIMENT 1

    # 20 hidden nodes
    nn = NeuralNetwork(train, test)
    hw1, ow1 = nn.train(0.1, 50, 20, output)
    nn.confusion(20, output, hw1, ow1)

    # 50 hidden nodes
    hw2, ow2 = nn.train(0.1, 50, 50, output)
    nn.confusion(50, output, hw2, ow2)

    # 100 hidden nodes
    hw3, ow3 = nn.train(0.1, 50, 100, output)
    nn.confusion(100, output, hw3, ow3)


    # EXPERIMENT 2

    # ETA 0.1, Hidden Units 100, 15000 rows
    nn2 = NeuralNetwork(train[15000:], test)
    hw4, ow4 = nn2.train(0.1, 50, 100, output)
    nn2.confusion(100, output, hw4, ow4)

    # ETA 0.1, Hidden Units 100, 30000 rows
    nn3 = NeuralNetwork(train[30000:], test)
    hw5, ow5 = nn3.train(0.1, 50, 100, output)
    nn3.confusion(100, output, hw5, ow5)
