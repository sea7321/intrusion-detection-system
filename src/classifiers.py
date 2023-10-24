"""
File: classifiers.py
Description: Decision tree classifier.
"""

# Standard Imports
import time

# Third-Party Imports
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from termcolor import colored


def calculate_rates(y, y_test, y_pred, dataset, classifier):
    """
    Calculates the rates of the dataset.
    :param y: (String[]) the columns in the dataset
    :param y_test: (String[]) the test data
    :param y_pred: (String[]) the predicted data
    :param dataset: (String) the dataset
    :return: (Tuple) Overall FPR and FNR
    """
    with open('./output/{}_{}_rates.txt'.format(classifier, dataset), 'w') as file:
        # calculate confusion matrix
        np.seterr(invalid='ignore')
        labels = list(set(y.to_list()))
        conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)

        # export confusion matrix to CSV file
        df_confusion = pd.DataFrame(conf_matrix, columns=labels, index=labels)
        df_confusion.to_csv('./output/{}_{}_confusion_matrix.csv'.format(classifier, dataset))

        # calculate FP, FN, TP, and TNs
        FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
        FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
        TP = np.diag(conf_matrix)
        TN = np.sum(np.concatenate(conf_matrix)) - (FP + FN + TP)

        # calculate rates
        FPR = FP / (FP + TN)
        FNR = FN / (FN + TP)
        sum_FP = sum_FP_TN = sum_FN = sum_FN_TP = 0

        # enumerate through each class
        for i, label in enumerate(labels):
            file.write(f"{label:<15}: \tFPR: {round(FPR[i] * 100, 2):<5.2f}%  ({FP[i]}/{FP[i] + TN[i]})"
                       f"\t\tFNR: {round(FNR[i] * 100, 2):<7.2f}%  ({FN[i]}/{FN[i] + TP[i]})\n")
            sum_FP += FP[i]
            sum_FP_TN += (FP[i] + TN[i])
            sum_FN += FN[i]
            sum_FN_TP += (FP[i] + TP[i])

        # calculate overall FPR and FNR
        overall_FPR = (sum_FP / sum_FP_TN) * 100
        overall_FNR = (sum_FN / sum_FN_TP) * 100
        file.write("\nOverall FPR: {:.4f} %\n".format(overall_FPR))
        file.write("Overall FNR: {:.4f} %\n".format(overall_FNR))

        return overall_FPR, overall_FNR


def decision_tree(dataset, classifier):
    """
    Classifies a given dataset based on the 'class' target variable.
    :param dataset: (String) the dataset name
    :return: None
    """
    # start the timer
    start = time.time()

    # load the training data
    print("\tLoading in the data...")
    data = pd.read_csv('../data/{}_training_data.csv'.format(dataset))

    # split the training data into features and the target variable
    X = data.drop('class', axis=1)
    y = data['class']

    # split the data into a training set and a testing set
    print("\tSplitting the data into a training and testing set...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # create the decision tree classifier
    classifier = DecisionTreeClassifier(random_state=42)

    # fit the classifier to the training data
    print("\tClassifying the training data...")
    classifier.fit(X_train, y_train)

    # make predictions on the test data
    print("\tMaking predictions on the testing data...")
    y_pred = classifier.predict(X_test)
    end = time.time()

    # calculate false positive and false negative rates
    rates = calculate_rates(y, y_test, y_pred, dataset, classifier)
    print(colored("\tOverall FPR: {:.4f} %".format(np.mean(rates[0])), "yellow"))
    print(colored("\tOverall FNR: {:.4f} %".format(np.mean(rates[1])), "yellow"))

    # calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(colored("\tAccuracy: {:.4f} %".format(accuracy * 100), "green"))
    print(colored("\tTime: {:.4f} seconds\n".format(end - start), "green"))


def multi_layer_perceptron(dataset, classifier):
    # start the timer
    start = time.time()

    # load the training data
    print("\tLoading in the data...")
    data = pd.read_csv('../data/{}_training_data.csv'.format(dataset))

    # split the training data into features and the target variable
    X = data.drop('class', axis=1)
    y = data['class']

    # split the dataset into training and testing sets
    print("\tSplitting the data into a training and testing set...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # create a Multilayer Perceptron (MLP) classifier
    mlp_classifier = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)

    # train the MLP classifier on the training data
    print("\tClassifying the training data...")
    mlp_classifier.fit(X_train, y_train)

    # make predictions on the test data
    print("\tMaking predictions on the testing data...")
    y_pred = mlp_classifier.predict(X_test)
    end = time.time()

    # calculate false positive and false negative rates
    rates = calculate_rates(y, y_test, y_pred, dataset, classifier)
    print(colored("\tOverall FPR: {:.4f} %".format(np.mean(rates[0])), "yellow"))
    print(colored("\tOverall FNR: {:.4f} %".format(np.mean(rates[1])), "yellow"))

    # calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(colored("\tAccuracy: {:.4f} %".format(accuracy * 100), "green"))
    print(colored("\tTime: {:.4f} seconds\n".format(end - start), "green"))
