from decision_tree import decision_tree
from preprocessing import preprocess


def main():
    # Preprocess data
    preprocess()

    # Run the decision tree classifier
    decision_tree()


if __name__ == "__main__":
    main()
