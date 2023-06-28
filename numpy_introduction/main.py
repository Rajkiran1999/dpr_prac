import numpy as np

from metrics import confusion_matrix, precision, recall, false_alarm_rate
from datasets import download_and_prepare
from recommender_system import MatrixFactorization


def main():
    np.random.seed(42)
    np.set_printoptions(precision=2, floatmode='fixed')

    # Part I
    print("------------------------------------------------")
    print("Part I - Confusion matrix")
    print("------------------------------------------------")

    y_true = np.random.randint(0, 2, 20)
    y_pred = np.random.randint(0, 2, 20)

    print("Unnormalized confusion matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    print("Matrix sum normalization:")
    cm = confusion_matrix(y_true, y_pred, normalize='all')
    print(cm)
    print("Row sum normalization:")
    cm = confusion_matrix(y_true, y_pred, normalize='pred')
    print(cm)
    print("Column sum normalization:")
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    print(cm)
    print(f"Precision: {precision(y_true, y_pred):.2f}, recall: {recall(y_true, y_pred):.2f}"
          f", false alarm rate: {false_alarm_rate(y_true, y_pred):.2f}")

    # Part II
    print("------------------------------------------------")
    print("Part II - Movie Recommender System")
    print("------------------------------------------------")

    X = download_and_prepare('movielens-small', '../datasets')
    matrixFactor = MatrixFactorization(X)
    r_hat = matrixFactor.fit()

if __name__ == "__main__":
    main()
