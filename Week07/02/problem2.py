# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import csv, sys, numpy as np
from decimal import Decimal

def import_data(inputFile):

    data, labels = [], []

    file = open(inputFile, 'rb')
    reader = csv.reader(file, delimiter=',')

    for row in reader:
        data.append([float(i) for i in row[:-1]])
        labels.append(float(row[-1]))

    file.close()

    return data, labels

def prepare_data(data, labels):
    # Normalise Data
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    labels = (labels - np.mean(labels, axis=0)) / np.std(labels, axis=0)

    # Insert Intersect Matrix of 1s.
    data = np.c_[np.ones(len(labels)), data]

    return data, labels

def gradient_descend(data, labels, alpha, iterations):
    betas = np.zeros(len(data[0]))
    risk = 0.0

    for i in xrange(100):
        risk = np.sum(np.square(np.dot(data, betas) - labels)) / (2 * len(labels))
        teste = np.dot(data, betas) - labels
        teste2 = np.transpose(( np.dot(data, betas) - labels ) * np.transpose(data))
        betas -= alpha * np.sum( np.transpose(( np.dot(data, betas) - labels ) * np.transpose(data)), 0 )/ len(labels)

    print risk
    return betas


def linear_regression(data, labels, alpha, iterations):

    # Initialize
    result = [alpha, iterations]

    # Find Betas by Gradient Descend
    betas = gradient_descend(data, labels, alpha, iterations)

    for i in range(len(betas)):
        result.append(Decimal(betas[i]))

    # Return results
    return result


# Main
if __name__ == '__main__':

    # Initialize Data
    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
    data, labels = import_data(sys.argv[1])
    results = []

    # Prepare Data
    normalized_data, normalized_labels = prepare_data(data, labels)

    # For each alpha, get the linear regression betas (EDX reported correct only for not normalized labels)
    for alpha in alphas:
        results.append(linear_regression(normalized_data, normalized_labels, alpha, 100))

    # My own parameters
    results.append(linear_regression(normalized_data, normalized_labels, 1, 5))

    # Write Output
    file = open(sys.argv[2], 'wb')
    writer = csv.writer(file, delimiter=',')

    for result in results:
        writer.writerow(result)

    file.close()

    plot_data = np.transpose(data)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(plot_data[0], plot_data[1], labels)
    ax.set_xlabel('Age (Years)')
    ax.set_ylabel('Weight (Kilograms)')
    ax.set_zlabel('Height (Meters)')

    x = np.linspace(-5, 6, 30)
    y = np.linspace(-5, 40, 30)

    X, Y = np.meshgrid(x, y)
    Z = float(results[-1][2]) + float(results[-1][3]) * X + float(results[-1][4]) * Y

    #ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z)

    plt.show()