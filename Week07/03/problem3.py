# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt

import numpy as np, scipy, sklearn, csv, sys
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn import svm, linear_model, tree, datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def import_data(inputFile):

    data, labels = [], []

    # Open file and remove the header
    file = open(inputFile, 'rU')
    file.readline()
    reader = csv.reader(file, delimiter=',')

    for row in reader:
        data.append([float(i) for i in row[:-1]])
        labels.append(float(row[-1]))

    file.close()

    return data, labels

def prepare_data(data, labels):

    data, labels = np.array(data), np.array(labels)

    return train_test_split(data, labels, test_size=0.4, stratify=labels)

def prepare_plot_data(data, labels):

    new_data = dict()

    for i in range(len(labels)):
        if new_data.has_key(labels[i]):
            new_data[labels[i]].append(data[i])
        else:
            new_data[labels[i]] = [data[i]]

    for plot_data in new_data:
        new_data[plot_data] = np.transpose(new_data[plot_data])

    return new_data

def svm_linear(train_data, test_data, train_labels, test_labels):

    parameters = {'kernel': ['linear'], 'C': [0.1, 0.5, 1, 5, 10, 50, 100]}

    clf = GridSearchCV(svm.SVC(), parameters, cv=5)

    clf.fit(train_data, train_labels)

    print 'svm_linear,{},{}'.format(clf.best_score_, clf.score(test_data, test_labels))

def svm_polynomial(train_data, test_data, train_labels, test_labels):

    parameters = {'kernel': ['poly'], 'C': [0.1, 1, 3], 'degree': [4, 5, 6], 'gamma': [0.1, 0.5]}

    clf = GridSearchCV(svm.SVC(), parameters, cv=5)

    clf.fit(train_data, train_labels)

    print 'svm_polynomial,{},{}'.format(clf.best_score_, clf.score(test_data, test_labels))

def svm_rbf(train_data, test_data, train_labels, test_labels):

    parameters = {'kernel': ['rbf'], 'C': [0.1, 0.5, 1, 5, 10, 50, 100], 'gamma': [0.1, 0.5, 1, 3, 6, 10]}

    clf = GridSearchCV(svm.SVC(), parameters, cv=5)

    clf.fit(train_data, train_labels)

    print 'svm_rbf,{},{}'.format(clf.best_score_, clf.score(test_data, test_labels))

def logisticRegression(train_data, test_data, train_labels, test_labels):

    parameters = {'C': [0.1, 0.5, 1, 5, 10, 50, 100]}

    clf = GridSearchCV(linear_model.LogisticRegression(), parameters, cv=5)

    clf.fit(train_data, train_labels)

    print 'logistic,{},{}'.format(clf.best_score_, clf.score(test_data, test_labels))

def knn(train_data, test_data, train_labels, test_labels):

    parameters = {'n_neighbors': range(1, 51), 'leaf_size': range(5, 61, 5)}

    clf = GridSearchCV(KNeighborsClassifier(), parameters, cv=5)

    clf.fit(train_data, train_labels)

    print 'knn,{},{}'.format(clf.best_score_, clf.score(test_data, test_labels))

def decision_tree(train_data, test_data, train_labels, test_labels):
    parameters = {'max_depth': range(1, 51), 'min_samples_split': range(2,11)}

    clf = GridSearchCV(tree.DecisionTreeClassifier(), parameters, cv=5)

    clf.fit(train_data, train_labels)

    print 'decision_tree,{},{}'.format(clf.best_score_, clf.score(test_data, test_labels))

def random_forest(data_train, data_test, labels_train, labels_test):
    parameters = {'max_depth': range(1, 51), 'min_samples_split': range(2, 11)}

    clf = GridSearchCV(RandomForestClassifier(), parameters, cv=5)

    clf.fit(train_data, train_labels)

    print 'random_forest,{},{}'.format(clf.best_score_, clf.score(test_data, test_labels))

# Main
if __name__ == '__main__':

    # Initialize Data
    data, labels = import_data(sys.argv[1])

    # Prepare data
    train_data, test_data, train_labels, test_labels = prepare_data(data, labels)
    train_plot = prepare_plot_data(data, labels)


    # Comment the function call to avoid time consuption
    #svm_linear(train_data, test_data, train_labels, test_labels)
    #svm_polynomial(train_data, test_data, train_labels, test_labels)
    #svm_rbf(train_data, test_data, train_labels, test_labels)
    #logisticRegression(train_data, test_data, train_labels, test_labels)
    knn(train_data, test_data, train_labels, test_labels)
    decision_tree(train_data, test_data, train_labels, test_labels)
    random_forest(train_data, test_data, train_labels, test_labels)

    # Write Output
    # file = open(sys.argv[2], 'w')
    # writer = csv.writer(file, delimiter=',')
    #
    # for result in results:
    #     writer.writerow(result)
    #
    # file.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(train_plot[0][0], train_plot[0][1], color='r')
    ax.scatter(train_plot[1][0], train_plot[1][1], color='b')
    ax.set_xlabel('A')
    ax.set_ylabel('B')

    plt.show()