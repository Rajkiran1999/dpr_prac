import time
from absl import logging
import gin

import numpy as np
import cv2
from skimage import exposure
from skimage.feature import hog
from skimage.transform import rescale, resize

import matplotlib.pyplot as plt


@gin.configurable
def read_file(file, grayscale=False):
    """
    Loads and normalizes image.

    Parameters
        file (string): image file path
        grayscale (bool): True converts the image to grayscale

    Returns:
        (ndarray): image
    """
    image = cv2.imread(file)
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.float32(image) / 255
    return image


def extract_hog_features(image, hog_config, visualize=False, transform_sqrt=True, feature_vector=False, multichannel=True):
    """
    Compute Histogram of Oriented Gradients for given image.

    Parameters:
        image (ndarray): image
        hog_config (dict): Example {"orientations": 8, "pixels_per_cell": (8, 8), "cells_per_block": (4, 4)}
        visualize (bool): whether to visualize the HOG features or not
        transform_sqrt (bool): whether to apply power law compression to normalize the image before processing
        feature_vector (bool): whether to return a feature vector or keep the dimensions (cells, cells, cells_per_block,
            cells_per_block, orientations)
        multichannel (bool): True for color images, False for grayscale images

    Returns:
         (ndarray): feature tensor/vector
    """
    fd = hog(image, orientations=hog_config["orientations"], pixels_per_cell=hog_config["pixels_per_cell"],
             cells_per_block=hog_config["cells_per_block"], visualize=visualize, transform_sqrt=transform_sqrt,
             feature_vector=feature_vector, multichannel=multichannel)
    return fd


@gin.configurable
def get_positive_features(files, template_size, hog_config):
    """
    Extract positive features from face crops.

    Loads face crop images, resizes them to (template_size, template_size), and extracts HOG features.

    Parameters:
        files (list): image file paths used to extract positive features
        template_size (int): the number of pixels spanned by a template used for feature extraction
        hog_config (dict): Example {"orientations": 8, "pixels_per_cell": (8, 8), "cells_per_block": (4, 4)}

    Returns:
         (ndarray): (positive) feature matrix with shape (n_samples, n_features)
    """
    logging.info(f"Extracting positive features...")

    # Visualize the hog features for the first image
    image = read_file(files[0])
    image = resize(image, output_shape=(template_size, template_size))
    _, hog_image = extract_hog_features(image, hog_config, visualize=True)
    visualize_hog(image, hog_image)

    # Extract hog features for all files
    start_time = time.time()
    # TODO: Extract positive features

    logging.debug(f"Feature extraction took {int(time.time() - start_time)} seconds.")

    return features


@gin.configurable
def get_negative_features(files, template_size, hog_config, scale, samples_per_scale):
    """
    Extract negative features from non-face scenes.

    Loads non-face scene images, randomly samples patches with size (template_size, template_size), and extracts HOG
    features. This is done at multiple scales. Starting at the original scale, the image is downscaled by the factor
    "scale" until the image size is smaller then the template size. "samples_per_scale" patches are extracted per sample
    scale.

    Parameters:
        files (list): image file paths used to extract negative features
        template_size (int): the number of pixels spanned by a template used for feature extraction
        hog_config (dict): Example {"orientations": 8, "pixels_per_cell": (8, 8), "cells_per_block": (4, 4)}
        scale (float): scale used to downscale image to extract negative features at different scales
        samples_per_scale (int): determines how many random patches are extracted per sample scale

    Returns:
         (ndarray): (negative) feature matrix with shape (n_samples, n_features)
    """
    logging.info(f"Extracting negative features...")

    # Extract hog features for all files
    start_time = time.time()
    # TODO: Extract negative features

    logging.debug(f"Feature extraction took {int(time.time() - start_time)} seconds.")

    return features


def visualize_hog(image, hog_image):
    """
    Creates a plot visualizing the image and its corresponding HOG features.

    Parameters:
        image (ndarray): image
        hog_image (ndarray): corresponding HOG features
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()