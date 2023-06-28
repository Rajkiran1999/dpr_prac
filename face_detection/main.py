import gin
import numpy as np
from absl import logging


from datasets import download_and_prepare
from classifier import FaceDetector
from feature_extraction import get_positive_features, get_negative_features


def main():
    # Positive mining
    files = download_and_prepare("utk-faces", "../datasets")
    pos_features = get_positive_features(files[::1])

    # Negative mining
    files = download_and_prepare("sun2012", "../datasets")
    neg_features = get_negative_features(files[::10]) # We will use only every 10th image, feel free to try out different values

    X = np.concatenate([pos_features, neg_features], axis=0)
    y = np.concatenate([np.ones(pos_features.shape[0]), -np.ones(neg_features.shape[0])], axis=0)

    # Train LinearSVM
    detector = FaceDetector()
    detector.train(X, y)

    # Evaluation
    files, labels = download_and_prepare("fddb", "../datasets")
    detector.evaluate(files[:100], labels[:100]) # We only use the first 100 images for testing to reduce time


if __name__ == "__main__":
    logging.set_verbosity(logging.DEBUG)
    gin.parse_config_file("config.gin")
    main()
