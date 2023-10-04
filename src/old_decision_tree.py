import math
from util import *
from old_preprocessing import normalize
import pandas as pd
import pickle as pkl
import time


def main():
    source_file = "../data/KDDTrain+.csv"
    dest_file = "../data/old_training_data.csv"

    ### PROCESS -- ~30 seconds
    print("preprocessing...")
    start = time.time()
    normalize(source_file, dest_file)
    end = time.time()
    print("Time to preprocess: ", end - start)

    ### TRAIN -- ~20 minutes to train with a depth of 25
    print("training...")
    training_data = pd.read_csv(dest_file)
    start = time.time()
    decision_tree = train(training_data, 25)
    end = time.time()
    print("Time to train: ", end - start)
    with open("tree1", "wb") as file:
        pkl.dump(decision_tree, file)

    ### PREDICT -- 1.7% FNR and 1.9% FPR with depth of 25
    print("evaluating...")
    with open("tree1", "rb") as file:
        tree = pkl.load(file)
        test_data = pd.read_csv(dest_file)
        num_negatives = 0
        f_neg = 0
        f_pos = 0
        num_correct = 0
        for index, row in test_data.iterrows():
            prediction = predict(tree.root, list(row))
            if row["class"] == "normal":
                num_negatives += 1
            if row["class"] != "normal" and prediction == "normal":
                f_neg += 1
            elif row["class"] == "normal" and prediction != "normal":
                f_pos += 1
            else:
                num_correct += 1
        print(f"Accuracy: {num_correct} / {len(test_data)}: {num_correct / len(test_data)}")
        print(f"False Negative Rate: {f_neg} / {num_negatives}: {f_neg / num_negatives}")
        print(f"False Alarm Rate: {f_pos} / {num_negatives}: {f_pos / num_negatives}")


def predict(node, record):
    if node.left is None and node.right is None:
        return "anomaly" if node.decision else "normal"
    return predict(node.right if record[node.attr] >= 0 else node.left, record)


def traverse(tree):
    traverse_h(tree.root, "")


def traverse_h(node, header):
    if node is None:
        return
    print(header, node.attr, node.decision)
    traverse_h(node.left, header + "--")
    traverse_h(node.right, header + "--")


def train(training_data, max_depth):
    return train_h(training_data, max_depth, 0)


def train_h(training_data, max_depth, depth):
    num_pos = len(list(filter(lambda cls: cls != "normal", training_data["class"].to_list())))
    decision = num_pos > (len(training_data) - num_pos)
    if len(training_data) == 0:
        return Tree(Node(None, decision))
    attr = find_best_attr(training_data)
    tree = Tree(Node(attr, decision))
    if max_depth == depth:
        return tree
    left = training_data[training_data[training_data.columns[attr]] < 0]
    right = training_data[training_data[training_data.columns[attr]] >= 0]
    tree.root.left = train_h(left, max_depth, depth + 1).root
    tree.root.right = train_h(right, max_depth, depth + 1).root

    return tree


def find_best_attr(training_data):
    highest_gain = 0
    attr = 0
    for index, col in enumerate(training_data.columns):
        if col == "Unnamed: 0":
            continue
        if training_data[col].dtype == "object":
            continue
        gain = information_gain(training_data, col)
        if gain > highest_gain:
            highest_gain = gain
            attr = index
    return attr


def information_gain(training_data, attr):
    num_pos = len(list(filter(lambda cls: cls != "normal", training_data["class"].to_list())))
    return boolean_entropy(num_pos / len(training_data)) - entropy_rem(training_data, attr)


def boolean_entropy(q):
    if q == 0 or q == 1:
        return 0
    return - (q * (math.log(q) / math.log(2)) + (1 - q) * (math.log(1 - q) / math.log(2)))


def entropy_rem(training_data, attr):
    return remainder(training_data, attr, 0) + remainder(training_data, attr, 1)


def remainder(training_data, attr, a):
    filt_data = training_data[training_data[attr] >= 0] if a == 0 else training_data[training_data[attr] < 0]
    num_pos = len(list(filter(lambda cls: cls != "normal", filt_data["class"].to_list())))
    num_neg = len(list(filter(lambda cls: cls == "normal", filt_data["class"].to_list())))
    if num_pos + num_neg == 0:
        return 0
    return ((num_pos + num_neg) / len(training_data)) * boolean_entropy(num_pos / (num_pos + num_neg))


if __name__ == "__main__":
    main()
