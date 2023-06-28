import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import ListedColormap


def plot_results(x_train, y_train, x_test, y_test, confusion_matrices, classifiers):

    nm, knn, gmm = classifiers
    titles = ['nearest mean classifier', f'{knn.k}-nearest neighbor classifier', 'Gaussian mixture model']
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for clf in range(len(classifiers)):

        # Plotting decision regions
        x_min, x_max = min(x_train[:, 0].min(), x_test[:, 0].min()) - 1, max(x_train[:, 0].max() + 1,
                                                                             x_test[:, 0].max() + 1)
        y_min, y_max = min(x_train[:, 1].min(), x_test[:, 1].min()) - 1, max(x_train[:, 1].max() + 1,
                                                                             x_test[:, 1].max() + 1)
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))

        z = classifiers[clf].predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)
        axes[0, clf].contourf(xx, yy, z, cmap=cm, alpha=0.8)

        # Subplot
        axes[0, clf].set_title(titles[clf])
        axes[0, clf].set_xlabel('x1')
        axes[0, clf].set_ylabel('x2')
        axes[0, clf].set_aspect('equal')

        # Training data
        axes[0, clf].scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright, marker='x', edgecolors='black',
                             label='training data')

        # Test data
        axes[0, clf].scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='black', label='test data')
        if clf == 0:
            axes[0, clf].scatter(nm.means[:, 0], nm.means[:, 1], c='yellow', marker='x', label='means')
        axes[0, clf].legend()

        # Confusion matrix
        axes[1, clf].matshow(confusion_matrices[clf], cmap=plt.cm.Blues)
        axes[1, clf].set_xlabel('True label')
        axes[1, clf].xaxis.set_label_position('top')
        axes[1, clf].set_ylabel('Predicted label')

        for i in range(2):
            for j in range(2):
                c = confusion_matrices[clf][i, j]
                axes[1, clf].text(i, j, str(c), va='center', ha='center')

    plt.tight_layout()
    plt.show()
