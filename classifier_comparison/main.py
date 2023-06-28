from absl import app
import numpy as np
from sklearn.metrics import confusion_matrix
import time

from datasets import two_moons_dataset, four_parallel_dataset, four_gaussian_dataset, circular_dataset
from classifiers import NearestMeanClassifier, KNearestNeighborClassifier, GaussianMixtureModelClassifier
from plot import plot_results


def main(argv):
    # load dataset
    x_train, y_train, x_test, y_test = two_moons_dataset(1000, 0.5)
    # x_train, y_train, x_test, y_test = four_parallel_dataset(1000, 0.5)
    # x_train, y_train, x_test, y_test = four_gaussian_dataset(1000, 0.5)
    # x_train, y_train, x_test, y_test = circular_dataset(1000, 0.5)


    # create classifiers
    nm = NearestMeanClassifier()
    knn = KNearestNeighborClassifier(k=3)
    gmm = GaussianMixtureModelClassifier(m=np.array([2, 2]))
    classifiers = [nm, knn, gmm]
    classifier_names = ['nearest mean classifier', f'{knn.k}-nearest neighbor classifier', 'Gaussian mixture model']

    # train classifiers
    print("----------------------Training----------------------")
    for classifier, name in zip(classifiers, classifier_names):
        start = time.time()
        classifier.fit(x_train, y_train)
        end = time.time()
        print(f"{name} (training): {end - start:.3f}s")

    # perform classification
    print("----------------------Inference---------------------")
    preds = []
    for classifier, name in zip(classifiers, classifier_names):
        start = time.time()
        preds.append(classifier.predict(x_test))
        end = time.time()
        print(f"{name} (inference): {end - start:.3f}s")

    # calculate metrics
    confusion_matrices = [np.transpose(confusion_matrix(y_test, pred)) for pred in preds]

    # plot results
    plot_results(x_train, y_train, x_test, y_test, confusion_matrices, classifiers)


if __name__ == '__main__':
    app.run(main)
