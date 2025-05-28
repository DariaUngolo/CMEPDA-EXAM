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
import numpy as np
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


    max_shape = np.max([img.shape for img in images], axis=0)

    padded_images = []
    for img in images:
        padding = []
        for i in range(3):  # for each dimension: x, y, z
            diff = max_shape[i] - img.shape[i]
            # Distribute padding equally before and after the image
            pad_before = diff // 2
            pad_after = diff - pad_before
            padding.append((pad_before, pad_after))
        # Apply zero padding using np.pad
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
    zoom_factor = np.random.uniform(*zoom_range)
    zoomed = zoom(volume, (zoom_factor, zoom_factor, zoom_factor), order=1)

    # Crop o pad
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

def augment_image_4d(volume_4d, target_shape):
    """
    Augmenta un'immagine 4D (x, y, z, 1).

    Parameters:
    -----------
    volume_4d : np.ndarray
        Immagine 4D (x, y, z, 1).
    target_shape : tuple of ints
        Dimensioni desiderate (x, y, z).

    Returns:
    --------
    np.ndarray
        Immagine augmentata 4D (x, y, z, 1).
    """
    volume = volume_4d[..., 0]  # Rimuovi la dimensione del canale
    volume = random_rotate(volume)
    volume = random_zoom_and_crop(volume, target_shape)
    return np.expand_dims(volume, axis=-1)  # Ripristina il canale

def augment_images_with_labels_4d(images, labels, target_shape, num_augmented_per_image):

    """

    Generates augmented images while preserving the associated original labels.

    Parameters:

    -----------
    images : np.ndarray
        Batch of original images (shape: [num_images, x, y, z, 1]).
    labels : np.ndarray
        Array of labels corresponding to the original images.
    target_shape : tuple of ints
        Desired dimensions (x, y, z).
    num_augmented_per_image : int
        Number of augmented images to generate for each original image.

    Returns:

    --------
    augmented_images : np.ndarray
        Batch of augmented images (num_augmented_images, x, y, z, 1).
    augmented_labels : np.ndarray
        Array of labels corresponding to the augmented images.

    """

    augmented_images = []
    augmented_labels = []

    for img, label in zip(images, labels):
        for _ in range(num_augmented_per_image):
            augmented_img = augment_image_4d(img, target_shape)
            augmented_images.append(augmented_img)
            augmented_labels.append(label)

    return np.array(augmented_images, dtype=np.float32), np.array(augmented_labels)



def preprocess_nifti_images(image_folder, atlas_path,metadata, roi_ids=(165, 166)):

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

    # Load atlas and get its data
    atlas = nib.load(atlas_path)
    atlas_data = atlas.get_fdata()

    # Determine the Z slice indices that include the specified ROIs
    z_indices = []
    for roi in roi_ids:
        z_indices.extend(np.unique(np.where(atlas_data == roi)[2]))
    z_min, z_max = min(z_indices), max(z_indices)

    # Load metadata (adjust path as necessary)
    metadata_csv = pd.read_csv(metadata,  sep='\t')
    meta_data = pd.DataFrame(metadata_csv)

    preprocessed_images = []

    # Iterate over NIFTI files in the folder
    for filename in os.listdir(image_folder):
        if filename.endswith(".nii") or filename.endswith(".nii.gz"):
            image_path = os.path.join(image_folder, filename)

            # Load NIFTI image and get data
            image = nib.load(image_path)
            image_data = image.get_fdata()

            # Extract the relevant Z range
            image_data = image_data[:, :, z_min-10:z_max + 10]

            # Remove black voxels (voxels with intensity 0)
            non_black_voxels = image_data > 0.000001 #DA CAPIRE CHE NUMERO METTERE
            cropped_data = image_data[np.ix_(
                np.any(non_black_voxels, axis=(1, 2)),
                np.any(non_black_voxels, axis=(0, 2)),
                np.any(non_black_voxels, axis=(0, 1))
            )]

            preprocessed_images.append(cropped_data)

    # Pad all images to the maximum shape
    images_padded, max_shape = pad_images(preprocessed_images)
    images = np.array(images_padded, dtype='float64')

    images = images[..., np.newaxis]

    # Extract group labels from metadata (assumed to be in the second column)
    group = meta_data.iloc[:, 1].map({'AD': 1, 'Normal': 0}).astype('float64').to_numpy()

    return images, group


def normalize_images_uniformly(images, global_min=None, global_max=None, min_val=0.0, max_val=1.0):

    """

    Normalize an array of images using a uniform range for all images.

    Parameters:

    -----------

    images : np.ndarray
        Array of images (4D: [num_images, x, y, z, 1]).
    global_min : float, optional
        Global minimum value used for normalization. If None, it is computed.
    global_max : float, optional
        Global maximum value used for normalization. If None, it is computed.
    min_val : float, optional
        Minimum value of the normalized range (default 0.0).
    max_val : float, optional
        Maximum value of the normalized range (default 1.0).

    Returns:

    --------

    np.ndarray
        Normalized images in the same 4D format.
    float, float
        Global minimum and maximum values used for normalization.

    """
    # Compute global min and max if not provided
    if global_min is None:
        global_min = np.min(images)
    if global_max is None:
        global_max = np.max(images)

    # Avoid division by zero
    if global_max == global_min:
        logger.warning("All images have the same value. Normalization is unnecessary.")
        return images, global_min, global_max

    # Normalize images using global min and max
    images_normalized = (images - global_min) / (global_max - global_min)
    images_normalized = images_normalized * (max_val - min_val) + min_val

    logger.info(f"Images normalized to range {min_val} to {max_val}.")
    return images_normalized



def split_data(images, group):

    """

    Splits the dataset into training, validation, and test sets.

    Parameters:

    -----------
    images : np.ndarray
        Array of images (e.g., DTI images).
    labels : np.ndarray
        Corresponding array of labels.

    Returns:

    --------
    x_train, y_train : np.ndarray
        Training images and labels.
    x_val, y_val : np.ndarray
        Validation images and labels.
    x_test, y_test : np.ndarray
        Test images and labels.

    """
    # Split dataset into train+val and test (80% train+val, 20% test)
    x_temp, x_test, y_temp, y_test = train_test_split(
        images, group, test_size=0.2, random_state=10
    )

    # Split train+val into train and validation (75% train, 25% val)
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp, test_size=0.25, random_state=20
    )

    return x_train, y_train, x_val, y_val, x_test, y_test


