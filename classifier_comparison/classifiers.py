import numpy as np
from sklearn.mixture import GaussianMixture


class NearestMeanClassifier:
    def __init__(self):
        self.means = np.array([[0, 0], [0, 0]])

    def fit(self, x_train, y_train):
        # calculate means
        # TODO: Extract the training samples of class 0 and calculate their mean
        x_train0 =
        mean0 =
        # TODO: Extract the training samples of class 1 and calculate their mean
        x_train1 =
        mean1 =
        self.means = np.stack((mean0, mean1))

    def predict(self, x_test):
        labels = np.array([], dtype=np.int32)

        for x in x_test:
            # TODO: calculate distances of test instance x to both means. Save result in a numpy array of the form
            #       [distance to mean 0, distance to mean 1]
            distances =
            # TODO: identify nearest mean
            label =
            # save the label in array which should finally contain the labels for all test instances
            labels = np.append(labels, label)

        return labels


class KNearestNeighborClassifier:
    def __init__(self, k):
        self.k = k
        self.x_train = np.array([])
        self.y_train = np.array([])

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        labels = np.array([], dtype=np.int32)

        for x in x_test:
            # TODO: calculate distances of test instance x to all training instances and save them all together in a
            #       numpy array
            distances =

            # TODO: sort the labels of the training instances according to their distance to x (the function argsort of
            #       numpy may be helpful)
            sorted_labels =

            # TODO: take first k labels
            knn_labels =

            # TODO: identify the most frequently occurring label (the function bincount of numpy may be helpful)
            label =

            # save the label in array which should finally contain the labels for all test instances
            labels = np.append(labels, label)

        return labels


class GaussianMixtureModelClassifier:
    def __init__(self, m):
        self.gmm0 = GaussianMixture(m[0])
        self.gmm1 = GaussianMixture(m[1])

    def fit(self, x_train, y_train):
        # fit one Gaussian mixture model to each class
        # TODO: Extract the training samples of class 0 and fit a Gaussian mixture model to them
        x_train0 =
        self.gmm0.

        # TODO: Extract the training samples of class 1 and fit a Gaussian mixture model to them
        x_train1 =
        self.gmm0.

    def predict(self, x_test):
        # maximum likelihood classification
        # TODO: Get the log-likelihoods of all test instances for both classes (meaning both Gaussian mixture models)
        log_likelihood0 =
        log_likelihood1 =
        # TODO: Use the log-likelihoods to compute the log-likelihood ratio for all test instances
        log_likelihood_ratio =
        labels = np.zeros_like(x_test[:, 0])
        # TODO: Set the labels to 1 where to the log-likelihood ratio test for ML decision decides for class 1
        labels[] = 1

        return labels
