"""
File: decision_tree.py
Description: Decision tree classifier.
"""

# Standard Imports
import time

# Third-Party Imports
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from termcolor import colored


def decision_tree(filename):
    """
    Classifies a given dataset based on the 'class' target variable.
    :param filename: (String) the dataset filename
    :return: None
    """
    # start the timer
    start = time.time()

    # load the training data
    print("\tLoading in the data...")
    data = pd.read_csv(filename)

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

    # calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    end = time.time()
    print(colored("\tAccuracy: {:.2f} %".format(accuracy * 100), "green"))
    print(colored("\tTime: {:.2f} seconds\n".format(end - start), "green"))

    # Calculate false positves and false negatives for each category
    labels = list(set(y.to_list()))
    CM = confusion_matrix(y_test, y_pred, labels=labels)

    FP = CM.sum(axis=0) - np.diag(CM)
    FN = CM.sum(axis=1) - np.diag(CM)
    TP = np.diag(CM)
    TN = np.sum(np.concatenate(CM)) - (FP + FN + TP)
    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)

    for i, label in enumerate(labels):
        print(f"{label}: \tFalse positives: {round(FPR[i] * 100, 2)}%, {FP[i]} / {FP[i] + TN[i]}, "
              f"\tFalse negatives: {round(FNR[i] * 100, 2)}%, {FN[i]} / {TP[i] + FN[i]}")
