import numpy as np

def confusion_matrix(y_true, y_pred, normalize=None):
    """Computes the confusion matrix from predictions and labels.

    The matrix columns represent the real labels and the rows represent the
    prediction labels. The confusion matrix is always a 2-D array of shape `[n_labels, n_labels]`,
    where `n_labels` is the number of valid labels for a given classification task. Both
    prediction and labels must be 1-D arrays of the same shape in order for this
    function to work.

    Parameters:
        y_true: 1-D array of real labels for the classification task.
        y_pred: 1-D array of predictions for a given classification.
        normalize: One of ['true', 'pred', 'all', None], corresponding to column sum, row sum, matrix sum, or no
                   normalization.

    Returns:
        A 2-D array with shape `[n_labels, n_labels]` representing the confusion
        matrix, where `n` is the number of possible labels in the classification
        task.
    """

    if normalize not in ['true', 'pred', 'all', None]:
        raise ValueError("normalize must be one of {'true', 'pred', 'all', None}")

    n_labels = # TODO (TASK 1)

    cm = np.zeros((n_labels, n_labels))
    # TODO (TASK 1)

    if normalize == 'true':
        cm = # TODO (TASK 1)
    elif normalize == 'pred':
        cm = # TODO (TASK 1)
    elif normalize == 'all':
        cm = # TODO (TASK 1)

    return cm


def precision(y_true, y_pred):
    return # TODO (TASK 2)


def recall(y_true, y_pred):
    return # TODO (TASK 2)


def false_alarm_rate(y_true, y_pred):
    return # TODO (TASK 2)
