import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def decision_tree():
    # Load the training data
    print("Loading in the data...")
    data = pd.read_csv('../data/training_data.csv')

    # Split the training data into features and the target variable
    X = data.drop('class', axis=1)
    y = data['class']

    # Split the data into a training set and a testing set
    print("Splitting the data into a training and testing set...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create the decision tree classifier
    classifier = DecisionTreeClassifier(random_state=42)

    # Fit the classifier to the training data
    print("Classifying the training data...")
    classifier.fit(X_train, y_train)

    # Make predictions on the test data
    print("Making predictions on the testing data...")
    y_pred = classifier.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: {}".format(accuracy * 100))

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
