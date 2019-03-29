import random
import sys


def perceptron(nodeCount, inputs, weights):

    output = 0
    for i in range(0, nodeCount):
        output += weights[i]*inputs[i]

    if output > 0:
        return 1
    else:
        return -1


def train(inputDataFilePath, outputDataFilePath, inputSize, alpha):

    # Initialize
    weights = []
    for i in range(0, inputSize + 1):
        weights.append(0.01 * random.randrange(-20, 20, 1))

    # Do until convergence
    outputDataFile = open(outputDataFilePath, "w")
    convergence = False
    while not convergence:
        inputDataFile = open(inputDataFilePath, "r")
        convergence = True

        for inputData in inputDataFile:
            # Prepare the Input Data
            inputs = [1]
            textInputs = inputData.split(",")
            for i in range(0, inputSize):
                inputs.append(int(textInputs[i]))

            # Get the label
            label = int(textInputs[inputSize])

            # Feed the data and get the prediction
            y = perceptron(inputSize + 1, inputs, weights)

            # Wrong prediction: Adjust the weights
            if y != label:
                convergence = False
                for i in range(0, inputSize + 1):
                    weights[i] += round(alpha * (label - y) * inputs[i], 2)
                break

        # Close File
        inputDataFile.close()

        # Print the Weights until now
        output=""
        for i in range(1, inputSize + 1):
            output += "%.2f" % weights[i] + ","
        output += "%.2f" % weights[0]
        outputDataFile.writelines(output+"\n")
        print output

train(sys.argv[1], sys.argv[2], 2, 0.1)