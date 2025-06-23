import os
import sys
from pathlib import Path
from loguru import logger

import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from keras import Sequential
from keras.layers import RandomRotation, RandomZoom, RandomCrop, RandomContrast
from scipy.ndimage import rotate, zoom

import tensorflow as tf
import random
import re


# Configure loguru logger
logger.remove()  # Remove default handler
logger.add(sys.stdout, level="DEBUG", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | {level} | {message}")


def sorting_key(filename):
    """
    Generate a sorting key from the filename based on group and number.

    The function extracts a group label ('CTRL' or 'AD') and a following number
    from the filename using a regex pattern. It returns a tuple `(group, number)`
    that can be used to sort filenames first by group, then numerically by the number.
    If the filename does not match the expected pattern, it returns a tuple
    that will place the filename at the end when sorting.

    Parameters
    ----------
    filename : str
        The filename string to parse.

    Returns
    -------
    tuple
        A tuple `(group, number)` where:
        - `group` is a string, either 'CTRL' or 'AD' if matched, otherwise an empty string.
        - `number` is an integer extracted from the filename, or `float('inf')` if no match.
    """
    match = re.search(r"(CTRL|AD)-(\d+)", filename)
    if match:
        group, number = match.groups()
        return (group, int(number))
    logger.warning(f"File {filename} does not match expected pattern.")
    return ("", float('inf'))

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
        # Apply zero padding
        padded_img = np.pad(img, padding, mode='constant', constant_values=0)
        padded_images.append(padded_img)

    logger.debug(f"All images padded to shape {max_shape}.")

    return np.array(padded_images)


def random_rotate_tf(volume):
    """
    Applies random 3D rotation to a volume using TensorFlow.
    Each direction is randomly rotated by a value between -2 and +2 degrees.

    Parameters
    ----------
    volume : tf.Tensor
        A 3D volume tensor (x, y, z).

    Returns
    -------
    tf.Tensor
        Randomly rotated 3D volume.

    """

    def rotate_fn(volume):
        """
        Helper function to perform rotation.

        """
        # Random angles for each axis
        angles = np.random.uniform(-0.017, 0.017 , size=3)

        rotated = rotate(volume, angles[0], axes=(1, 2), reshape=False)
        rotated = rotate(rotated, angles[1], axes=(0, 2), reshape=False)
        rotated = rotate(rotated, angles[2], axes=(0, 1), reshape=False)
        return rotated

    volume = tf.numpy_function(rotate_fn, [volume], tf.float32)
    return volume



def random_zoom_and_crop_tf(volume, target_shape, zoom_range=(0.99, 1.01)):
    """

    Randomly zooms into a 3D volume and adjusts its dimensions to match the target shape.

    Parameters
    ----------
    volume : tf.Tensor
        A 3D volume tensor (x, y, z).
    target_shape : tuple of ints
        Desired dimensions (x, y, z).
    zoom_range : tuple of floats, optional
        Range of zoom factors (default is (0.99, 1.01)).

    Returns
    -------
    tf.Tensor
        Transformed volume tensor with the target dimensions.

    """
    def zoom_crop_fn(volume):
        """
        Helper function for zooming and cropping.

        """
        zoom_factor = np.random.uniform(*zoom_range)
        zoomed = zoom(volume, (zoom_factor, zoom_factor, zoom_factor), order=1)

        # Determine the cropping and padding
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

    volume = tf.numpy_function(zoom_crop_fn, [volume], tf.float32)
    return volume

def add_noise_tf(volume, noise_factor=0.001):
    """

    Adds random Gaussian noise to a 3D volume.

    Parameters
    ----------
    volume : tf.Tensor
        Input 3D volume tensor.
    noise_factor : float
        Standard deviation of the Gaussian noise.

    Returns
    -------
    tf.Tensor
        Noisy 3D volume tensor.

    """
    noise = tf.random.normal(tf.shape(volume), mean=0.0, stddev=noise_factor , dtype=volume.dtype)
    return tf.clip_by_value(volume + noise, 0.0, 1.0)


def random_intensity_tf(volume, factor=0.001):
    """
    Randomly modifies the intensity of a 3D volume.

    Parameters:
    -----------
    volume : np.ndarray
        3D volume to be modified.
    factor : float
        Range for the intensity scaling factor (default: 0.1).

    Returns:
    --------
    np.ndarray
        Volume with modified intensity.
    """

    scale = tf.random.uniform([], 1 - factor, 1 + factor, dtype=volume.dtype)
    return tf.clip_by_value(volume * scale, 0.0, 1.0)

def apply_all_transforms(volume, target_shape):
    """
    Applies all transformations one by one to the volume.

    Parameters:
    -----------
    volume : tf.Tensor
        3D volume tensor.
    target_shape : tuple of ints
        Desired output shape.

    Returns:
    --------
    List of tf.Tensor
        List of transformed volumes (one per transformation).
    """
    transforms = [
        #lambda vol: random_rotate_tf(vol),
        lambda vol: random_zoom_and_crop_tf(vol, target_shape),
        #lambda vol: add_noise_tf(vol),
        lambda vol: random_intensity_tf(vol)
    ]

    transformed_volumes = []
    for transform in transforms:
        transformed_volumes.append(transform(volume))
    return transformed_volumes

def augment_images_with_labels_4d(images, labels, target_shape):
    """
    Generates augmented images by including original images and all transformations applied to each original image.

    Parameters:
    -----------
    images : np.ndarray
        Original images (shape: [num_images, x, y, z, 1]).
    labels : np.ndarray
        Corresponding labels.
    target_shape : tuple of ints
        Desired output shape (x, y, z).

    Returns:
    --------
    augmented_images : np.ndarray
        Augmented images (num_images * (1 + num_transforms), x, y, z, 1).
    augmented_labels : np.ndarray
        Corresponding labels.
    """
    # Initialize lists to hold the augmented images and their corresponding labels
    augmented_images = []
    augmented_labels = []

    # Loop through each image and its label in the dataset
    for img, label in zip(images, labels):

        augmented_images.append(img)
        augmented_labels.append(label)

        # Convert the first channel of the image to a TensorFlow tensor
        img_tensor = tf.convert_to_tensor(img[..., 0])
        transformed_imgs = apply_all_transforms(img_tensor, target_shape)

        for transformed_img in transformed_imgs:
            transformed_img = tf.expand_dims(transformed_img, axis=-1)

            # Append the transformed image and the original label to the augmented lists
            augmented_images.append(transformed_img.numpy())
            augmented_labels.append(label)

    return np.array(augmented_images, dtype=np.float32), np.array(augmented_labels)


def preprocessed_images(image_folder, atlas_path, roi_ids=(165, 166)):

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
    try:
        atlas = nib.load(atlas_path)
        atlas_data = atlas.get_fdata()
        logger.info("Atlas successfully loaded.")
    except Exception as e:
        logger.error(f"Failed to load atlas: {e}")
        raise

    # Determine the Z slice indices that include the specified ROIs
    z_indices = []
    for roi in roi_ids:
        z_indices.extend(np.unique(np.where(atlas_data == roi)[2]))
    z_min, z_max = min(z_indices), max(z_indices)



    preprocessed_images = []



    file_list = [f for f in os.listdir(image_folder) if f.endswith(".nii") or f.endswith(".nii.gz")]


    sorted_file_list = sorted(file_list, key=sorting_key)

    # Iterate over NIFTI files in the folder
    for filename in sorted_file_list:
        if filename.endswith(".nii") or filename.endswith(".nii.gz"):
            image_path = os.path.join(image_folder, filename)

            # Load NIFTI image and get data
            image = nib.load(image_path)
            image_data = image.get_fdata()

            # Extract the relevant Z range
            image_data = image_data[:, :, z_min-10:z_max + 10]

            # Remove black voxels (voxels with intensity 0)
            non_black_voxels = image_data > 0.000001
            cropped_data = image_data[np.ix_(
                np.any(non_black_voxels, axis=(1, 2)),
                np.any(non_black_voxels, axis=(0, 2)),
                np.any(non_black_voxels, axis=(0, 1))
            )]

            preprocessed_images.append(cropped_data)

    # Pad all images to the maximum shape
    images_padded = pad_images(preprocessed_images)
    images = np.array(images_padded, dtype='float64')
    logger.info("All images preprocessed and padded.")
    images = images[..., np.newaxis]


    return images

def preprocessed_images_group(image_folder, atlas_path, metadata, roi_ids=(165, 166)):

    logger.info("Starting image preprocessing.")
    # Get preprocessed images
    images = preprocessed_images(image_folder, atlas_path, roi_ids)

    # Load metadata (adjust path as necessary)
    metadata_csv = pd.read_csv(metadata,  sep='\t')
    meta_data = pd.DataFrame(metadata_csv)

    # Extract group labels from metadata (assumed to be in the second column)
    group = meta_data.iloc[:, 1].map({'AD': 1, 'Normal': 0}).astype('float64').to_numpy()

    file_list = [f for f in os.listdir(image_folder) if f.endswith(".nii") or f.endswith(".nii.gz")]

    logger.debug(f"Preprocessed image dimensions: {images.shape}")
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
    logger.info(f"Normalized intensity of voxel")

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

    Splits the dataset into training(70%), validation(15%), and test sets(15%)

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
    logger.info("Splitting the dataset into train, validation, and test sets (70-15-15).")

    # Split dataset into train+val and test (85% train+val, 15% test)
    x_temp, x_test, y_temp, y_test = train_test_split(
        images, group, test_size=0.15,random_state=10
    )

    # Split train+val into train and validation (82% train, 18% val)
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp, test_size=0.1765, random_state=10
    )

    return x_train, y_train, x_val, y_val, x_test, y_test

def adjust_image_shape(image, target_shape):
    """
    Adjust the 4D image array to match the target shape by cropping or padding.

    Parameters
    ----------
    image : np.ndarray
        4D array of shape (N, X, Y, Z, 1) where N is batch size.
    target_shape : tuple
        Desired spatial shape (X, Y, Z).

    Returns
    -------
    np.ndarray
        Adjusted image array with shape (N, *target_shape, 1).
    """
    adjusted_images = []
    for img in image:
        # Remove channel dim temporarily
        img_3d = img[..., 0]

        # Crop if too large
        crop_slices = []
        for dim_idx in range(3):
            current_size = img_3d.shape[dim_idx]
            desired_size = target_shape[dim_idx]
            if current_size > desired_size:
                start = (current_size - desired_size) // 2
                end = start + desired_size
                crop_slices.append(slice(start, end))
            else:
                crop_slices.append(slice(0, current_size))

        img_cropped = img_3d[crop_slices[0], crop_slices[1], crop_slices[2]]

        # Pad if too small
        pad_width = []
        for dim_idx in range(3):
            current_size = img_cropped.shape[dim_idx]
            desired_size = target_shape[dim_idx]
            total_pad = max(desired_size - current_size, 0)
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
            pad_width.append((pad_before, pad_after))

        img_padded = np.pad(img_cropped, pad_width, mode='constant', constant_values=0)

        # Add channel dim back
        img_final = img_padded[..., np.newaxis]
        adjusted_images.append(img_final)

    return np.array(adjusted_images, dtype='float64')

