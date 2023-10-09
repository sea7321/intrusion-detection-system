"""
File: intrusion_detection_system.py
Description: Main module that classifies the NSL-KDD dataset.
"""

# Third-Party Imports
from colorama import init
from termcolor import colored

# Local Imports
from decision_tree import decision_tree
from preprocessing import preprocess


def main():
    # initialize colorama
    init()

    # preprocess data
    print(colored("[*] Preprocessing data...", "blue"))
    preprocess()

    # run the decision tree classifier for misuse detection
    print(colored("[*] Running the model for misuse-based dataset...", "blue"))
    decision_tree("misuse")

    # run the decision tree classifier for anomaly detection
    print(colored("[*] Running the model against anomaly-based dataset...", "blue"))
    decision_tree("anomaly")


if __name__ == "__main__":
    main()
