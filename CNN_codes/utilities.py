import os
import sys
from pathlib import Path
import logging

import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from keras import Sequential
from keras.layers import RandomRotation, RandomZoom, RandomCrop, RandomContrast
from scipy.ndimage import rotate, zoom
from skimage import exposure

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def pad_images(images):
    
    """
    Pads a list of 3D images to the same shape by adding zero-padding.

    Parameters:
    -----------
    images : list of np.ndarray
        List of 3D numpy arrays with varying shapes.

    Returns:
    --------
    padded_images : np.ndarray
        4D numpy array where each 3D image is padded to the maximum shape.
    max_shape : tuple
        The maximum shape found across all images.
        
    """
    logger.debug("Padding images to the same size.")
    max_shape = np.max([img.shape for img in images], axis=0)

    padded_images = []
    for img in images:
        padding = []
        for i in range(3):  # For each dimension: x, y, z
            diff = max_shape[i] - img.shape[i]
            pad_before = diff // 2
            pad_after = diff - pad_before
            padding.append((pad_before, pad_after))
        padded_img = np.pad(img, padding, mode='constant', constant_values=0)
        padded_images.append(padded_img)

    logger.debug(f"All images padded to shape {max_shape}.")
    return np.array(padded_images), max_shape

def random_rotate(volume):
    
    """
    Randomly rotates a 3D volume around the x, y, z axes.

    Parameters:
    -----------
    volume : np.ndarray
        3D volume (x, y, z).

    Returns:
    --------
    np.ndarray
        Rotated volume.
        
    """
    logger.debug("Applying random rotation to the volume.")
    angles = np.random.uniform(-0.5, 0.5, size=3)
    rotated = rotate(volume, angles[0], axes=(1, 2), reshape=False)
    rotated = rotate(rotated, angles[1], axes=(0, 2), reshape=False)
    rotated = rotate(rotated, angles[2], axes=(0, 1), reshape=False)
    return rotated

def random_zoom_and_crop(volume, target_shape, zoom_range=(0.8, 1.2)):
    
    """
    Randomly zooms into a 3D volume and crops or pads it to a target shape.

    Parameters:
    -----------
    volume : np.ndarray
        3D volume (x, y, z).
    target_shape : tuple of ints
        Desired dimensions (x, y, z).
    zoom_range : tuple of floats, optional
        Range of zoom factors (default is (0.8, 1.2)).

    Returns:
    --------
    np.ndarray
        Transformed volume with the target dimensions.
        
    """
    logger.debug("Applying random zoom and cropping the volume.")
    zoom_factor = np.random.uniform(*zoom_range)
    zoomed = zoom(volume, (zoom_factor, zoom_factor, zoom_factor), order=1)

    output = np.zeros(target_shape, dtype=zoomed.dtype)
    min_shape = np.minimum(zoomed.shape, target_shape)

    crop_start = [(s - ts) // 2 for s, ts in zip(zoomed.shape, min_shape)]
    crop_end = [start + ts for start, ts in zip(crop_start, min_shape)]
    pad_start = [(ts - ms) // 2 for ms, ts in zip(min_shape, target_shape)]
    pad_end = [start + ms for start, ms in zip(pad_start, min_shape)]

    output[pad_start[0]:pad_end[0], pad_start[1]:pad_end[1], pad_start[2]:pad_end[2]] = zoomed[
        crop_start[0]:crop_end[0], crop_start[1]:crop_end[1], crop_start[2]:crop_end[2]
    ]
    return output

def preprocess_nifti_images(image_folder, atlas_path, meta_data, roi_ids=(165, 166)):
    
    """
    Processes NIFTI images by extracting a Z range containing specified ROIs,
    removing black voxels, and padding images to uniform size.

    Parameters:
    -----------
    image_folder : str
        Path to the folder containing NIFTI images.
    atlas_path : str
        Path to the NIFTI atlas file.
    roi_ids : tuple of ints, optional
        Tuple of ROI IDs to include when determining Z range (default is (165, 166)).

    Returns:
    --------
    images : np.ndarray
        4D numpy array of preprocessed and padded 3D images ready for CNN input.
    group : np.ndarray
        Array of corresponding group labels loaded from metadata.
    """
    
    logger.debug("Loading atlas and determining Z range with specified ROIs.")
    atlas = nib.load(atlas_path)
    atlas_data = atlas.get_fdata()

    z_indices = []
    for roi in roi_ids:
        z_indices.extend(np.unique(np.where(atlas_data == roi)[2]))
    z_min, z_max = min(z_indices), max(z_indices)
    logger.debug(f"Z range determined: {z_min}-{z_max}.")

    preprocessed_images = []

    for filename in os.listdir(image_folder):
        if filename.endswith(".nii") or filename.endswith(".nii.gz"):
            logger.debug(f"Processing file: {filename}.")
            image_path = os.path.join(image_folder, filename)
            image = nib.load(image_path)
            image_data = image.get_fdata()

            image_data = image_data[:, :, z_min-10:z_max+10]

            non_black_voxels = image_data > 0.000001
            cropped_data = image_data[np.ix_(
                np.any(non_black_voxels, axis=(1, 2)),
                np.any(non_black_voxels, axis=(0, 2)),
                np.any(non_black_voxels, axis=(0, 1))
            )]
            preprocessed_images.append(cropped_data)

    images_padded, max_shape = pad_images(preprocessed_images)
    images = np.array(images_padded, dtype='float64')
    images = images[..., np.newaxis]

    group = meta_data.iloc[:, 1].map({'AD': 1, 'Normal': 0}).astype('float64').to_numpy()

    logger.debug("Preprocessing completed. Returning images and group labels.")
    return images, group

def split_data(images, group):
    
    """
    Splits the dataset into training, validation, and test sets.

    Parameters:
    -----------
    images : np.ndarray
        Array of images (e.g., DTI images).
    group : np.ndarray
        Corresponding array of group labels.

    Returns:
    --------
    x_train, y_train : np.ndarray
        Training images and labels.
    x_val, y_val : np.ndarray
        Validation images and labels.
    x_test, y_test : np.ndarray
        Test images and labels.
        
    """
    
    logger.debug("Splitting data into training, validation, and test sets.")
    x_temp, x_test, y_temp, y_test = train_test_split(
        images, group, test_size=0.2, random_state=10
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp, test_size=0.25, random_state=20
    )
    logger.debug("Data split completed.")
    return x_train, y_train, x_val, y_val, x_test, y_test
