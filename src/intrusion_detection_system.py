"""
File: intrusion_detection_system.py
Description: Main module that classifies the NSL-KDD dataset.
"""
# Standard Imports
import time
import threading
import argparse

# Third-Party Imports
import psutil
from colorama import init
from termcolor import colored

# Local Imports
from preprocessing import preprocess
from train import decision_tree, multilayer_perceptron
from classify import classify

# Global flag to signal the second thread to stop
stop_event = threading.Event()


def monitoring():
    """
    Monitors CPU and memory usage.
    :return: None
    """
    cpu_percent = memory_used = count = 0
    memory_info = psutil.virtual_memory()
    memory_total = memory_info.total / (1024 * 1024)

    while not stop_event.is_set():
        # get CPU usage percentage
        cpu_percent += psutil.cpu_percent()

        # get memory usage
        memory_info = psutil.virtual_memory()
        memory_used += memory_info.used / (1024 * 1024)

        # get cpu and memory usage every 1 second
        count += 1
        time.sleep(1)

    # calculate average cpu and memory usage
    average_cpu = cpu_percent / count
    average_memory = memory_used / count

    print("Average CPU Usage: {:.2f} %".format(average_cpu))
    print("Average Memory Usage: {} MB / {} MB ({:.2f} %)".format(round(average_memory), round(memory_total),
                                                                  (average_memory / memory_total) * 100))


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", action='store_true', help="load in classifiers from files")
    parser.add_argument("-c", "--classify", action='store', nargs=1, help="load in classifiers from files")
    args = parser.parse_args()

    # initialize colorama
    init()

    if args.train:
        # preprocess data
        print(colored("[*] Preprocessing data...", "blue"))
        preprocess()

        # run the decision tree classifier for misuse detection
        print(colored("[*] Running the decision tree classifier against the misuse-based dataset...", "blue"))
        decision_tree("misuse", "dt")

        # run the decision tree classifier for anomaly detection
        print(colored("[*] Running the decision tree classifier against the anomaly-based dataset...", "blue"))
        decision_tree("anomaly", "dt")

        # run the decision tree classifier for misuse detection
        print(colored("[*] Running the multilayer perceptron classifier against the misuse-based dataset...", "blue"))
        multilayer_perceptron("misuse", "mlp")

        # run the decision tree classifier for anomaly detection
        print(colored("[*] Running the multilayer perceptron classifier against the anomaly-based dataset...", "blue"))
        multilayer_perceptron("anomaly", "mlp")

        # signal the monitoring thread to stop
        stop_event.set()

    if args.classify:
        # run the decision tree classifier for misuse detection
        print(colored("[*] Running the misuse decision tree model against {}...".format(args.classify[0]), "blue"))
        classify("misuse", "dt", args.classify[0])

        # run the decision tree classifier for misuse detection
        print(colored("[*] Running the anomaly decision tree model against {}...".format(args.classify[0]), "blue"))
        classify("anomaly", "dt", args.classify[0])

        # run the decision tree classifier for misuse detection
        print(colored("[*] Running the misuse decision tree model against {}...".format(args.classify[0]), "blue"))
        classify("misuse", "mlp", args.classify[0])

        # run the decision tree classifier for misuse detection
        print(colored("[*] Running the anomaly decision tree model against {}...".format(args.classify[0]), "blue"))
        classify("anomaly", "mlp", args.classify[0])

        # signal the monitoring thread to stop
        stop_event.set()


if __name__ == "__main__":
    # Create and start the main thread
    thread1 = threading.Thread(target=main)
    thread1.start()

    # Create and start the monitoring thread
    thread2 = threading.Thread(target=monitoring)
    thread2.start()

    # Wait for both threads to finish
    thread1.join()
    thread2.join()
