from absl import logging

from sklearn.metrics import accuracy_score

from datasets import download_and_prepare
from classifier import SpamClassifier


def main():
    X_train, X_test, y_train, y_test = download_and_prepare("sms_spam", "../datasets")

    spam_classifier = SpamClassifier()
    spam_classifier.train(X_train, y_train)

    logging.info(f"Train Accuracy: {accuracy_score(y_train, spam_classifier.predict(X_train))}")
    logging.info(f"Test Accuracy: {accuracy_score(y_test, spam_classifier.predict(X_test))}")
    

if __name__ == "__main__":
    logging.set_verbosity(logging.DEBUG)
    main()
