"""
File: classify.py
Description: Classifies attack data on pre-saved models.
"""
# Standard Imports
import time

# Third-Party Imports
import pandas as pd
import pickle
from termcolor import colored


def classify(dataset, classifier_name, testing_filename):
    """
    Classifies an attack with the provided pre-saved classifier model.
    :param dataset: (String) the dataset name
    :param classifier_name: (String) the classifier name
    :param testing_filename: (String) the testing filename
    :return: None
    """
    # start the timer
    start = time.time()

    # load in the training model
    classifier = pickle.load(open("./models/{}_{}_model.pickle".format(classifier_name, dataset), "rb"))

    # load the testing data
    print("\tLoading in the data...")
    data = pd.read_csv(testing_filename)

    # split the testing data into features and the target variable
    X_test = data.drop('class', axis=1)
    y_test = data['class']

    # make predictions on the test data
    print("\tMaking predictions on the testing data...")
    y_pred = classifier.predict(X_test)
    end = time.time()

    # attack prediction
    if y_pred[0] != 'normal':
        print(colored("\tLikely an attack ({})".format(y_pred[0]), "red"))
    else:
        print(colored("\tNot likely an attack ({})".format(y_pred[0]), "yellow"))

    # calculate the accuracy of the model
    print(colored("\tTime: {:.4f} seconds\n".format(end - start), "green"))
