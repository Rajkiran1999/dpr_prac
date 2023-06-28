import gin
from absl import logging
import time
import os

from sklearn.svm import LinearSVC
from skimage.transform import rescale

import numpy as np
import cv2

from feature_extraction import extract_hog_features, read_file
from visualizations import visualize_detections
from postprocessing import non_max_suppression
from metrics import PrecisionRecallCurve


@gin.configurable
class FaceDetector(object):
    """
    Multi-scale sliding window face detector.

    Parameters:
        template_size (int): the number of pixels spanned by a template used for feature extraction
        hog_config (dict): Example {"orientations": 8, "pixels_per_cell": (8, 8), "cells_per_block": (4, 4)}
        confidence_threshold (float): samples with a confidence higher than confidence_threshold are considered detections
        iou_threshold (float): determines at what intersection over union a detection considered a match
        step_size (int): step size for sliding window in pixels
        pyramid_scale (float): scale used for downscaling the image within the multi-scale detector
        regularization_param (float): regularization parameter used for SVM

    """
    def __init__(self, template_size, hog_config, confidence_threshold, iou_threshold, step_size, pyramid_scale,
                 regularization_param):
        super(FaceDetector, self).__init__()
        self.classifier = LinearSVC(C=regularization_param, max_iter=1e5, class_weight="balanced")
        self.template_size = template_size
        self.hog_config = hog_config
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.step_size = step_size
        self.pyramid_scale = pyramid_scale


    def train(self, X, y):
        """
        Training method

        Parameters:
            X (ndarray): feature matrix
            y (ndarray): corresponding labels
        """
        logging.info(f"Starting training...")
        start_time = time.time()
        # TODO: Fit classifier to training data

        logging.debug(f"Training took {int(time.time() - start_time)} seconds.")


    def detect(self, image):
        """
        Run detector

        Starting at the original scale, the image is downscaled by the factor "pyramid_scale" until the image size is
        smaller than the template size. For each scale, HOG features for all patches are computed. The trained SVM
        predicts a confidence score for each patch, if the score is larger than the confidence threshold, it is treated
        as a detection. To remove duplicate detections, non-maximum suppression is performed.

        Parameters:
            image (ndarray): image for detecting faces

        Returns:
            (list, list): list of detection boxes [x_min, y_min, x_max, y_max], list of corresponding confidences
        """

        boxes = []
        confidences = []

        # TODO: Implement Face Detector

        # Perform non maximum suppression to get rid of duplicate detections
        boxes, confidences = non_max_suppression(boxes, confidences)

        return boxes, confidences


    def evaluate(self, files, labels):
        """
        Evaluates detector on test data.

        Plots precision recall curve and computes average precision.

        Parameters:
            files (list): image file paths
            labels (list): ground truth boxes
        Returns:
            (float): average precision
        """

        logging.info(f"Starting evaluation...")
        start_time = time.time()

        pr = PrecisionRecallCurve()

        for file, gt_boxes in zip(files, labels):
            y_true, y_score = self._evaluate_single_image(file, gt_boxes)
            pr.update_states(y_true, y_score, len(gt_boxes))

        ap = pr.result()

        logging.info(f"Average precision of {ap} achieved.")
        logging.debug(f"Evaluation took {int(time.time() - start_time)} seconds.")

        return ap


    def _evaluate_single_image(self, file, gt_boxes):
        """
        Performs evaluation on a single image.

        Loads image and runs detection. Non maximum suppression is used to remove duplicate detections.
        The intersection over union (IoU) is computed for every predicted box with every ground truth box (in our case
        the ground truth boxes are actually ellipses). In case the iou is larger than the iou_threshold, the detection
        is marked as a true positive.

        Parameters:
            file (list): image file path
            gt_boxes (list): ground truth boxes

        Returns:
            (list, list): true positives with corresponding confidence scores
        """

        def iou(box, ellipse, shape):
            """Intersection over Union"""

            box_mask = np.zeros(shape)
            box_mask = cv2.rectangle(box_mask, (box[0], box[1]), (box[2], box[3]), 1, -1)
            ellipse_mask = np.zeros(shape)
            ellipse_mask = cv2.ellipse(ellipse_mask, (ellipse[3], ellipse[4]), (ellipse[0], ellipse[1]), ellipse[2],
                                       0, 360, 1, -1)

            union = np.sum(np.maximum(box_mask, ellipse_mask))
            intersection = np.sum(np.minimum(box_mask, ellipse_mask))
            iou = intersection / union

            return iou

        logging.debug(f"Evaluation of {file}")
        image = read_file(file)

        boxes, confidences = self.detect(image)

        # Check if detections are true positives or not
        true_positives = [0] * len(boxes)
        gt_boxes_marked = [False] * len(gt_boxes)
        shape = image.shape[:2]
        for idx, box in enumerate(boxes):
            for gt_idx, gt_box in enumerate(gt_boxes):
                if iou(box, gt_box, shape) >= self.iou_threshold:
                    if not gt_boxes_marked[gt_idx]:
                        true_positives[idx] = 1
                        gt_boxes_marked[gt_idx] = True
                    break

        # Visualize detections
        visualize_detections(image, boxes, confidences, gt_boxes, true_positives,
                             save_path=os.path.join("results", "images_nms", file.split(os.sep)[-1]))
        return true_positives, confidences

