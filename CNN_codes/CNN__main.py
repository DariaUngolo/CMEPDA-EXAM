import sys
from pathlib import Path
import os
import argparse
import logging

from CNN_class import MyCNNModel
from utilities import preprocess_nifti_images, split_data, augment_images_with_labels_4d, normalize_images_uniformly

import argparse

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """
    Parse command-line arguments for the CNN training script.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
        
    """
    parser = argparse.ArgumentParser(
        description="Script for training a CNN model on NIfTI images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # === Input paths ===
    parser.add_argument(
        '--image_folder', type=str, required=True,
        help="Path to the folder containing NIfTI images."
    )
    parser.add_argument(
        '--atlas_path', type=str, required=True,
        help="Path to the NIfTI atlas file."
    )
    parser.add_argument(
        '--metadata', type=str, required=True,
        help="Path to CSV metadata file containing labels."
    )


    # === Model parameters ===

    parser.add_argument(
        '--epochs', type=int, default=50,
        help="Number of training epochs (default: %(default)s)."
    )
    parser.add_argument(
        '--batchsize', type=int, default=20,
        help="Batch size for training (default: %(default)s)."
    )

    return parser.parse_args()



def main(args):
    """
    Main function that performs preprocessing, data augmentation,
    data splitting, and CNN model training.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
        
    """

    roi_ids=(165,166)
    num_augmented_per_image = 2


    # Image preprocessing
    logger.info("Starting image preprocessing.")
    images, labels = preprocess_nifti_images(args.image_folder, args.atlas_path, args.metadata, tuple(roi_ids))
    logger.debug(f"Preprocessed image dimensions: {images.shape}")
    logger.debug(f"Preprocessed label dimensions: {labels.shape}")
    # Print first few labels
    logger.info(f"Sample labels: {labels[:10]}")  # Print the first 10 labels
    logger.info(f"Sample labels: {labels[-10:]}")  # Print the last 10 labels



    target_shape = images.shape[1:4]
    input_shape = images.shape[1:]

    logger.info(f"Using target shape for augmentation derived from training data: {target_shape}")
    logger.info(f"Using input shape for the CNN model: {input_shape}")

    # Data augmentation
    logger.info("Starting image augmentation.")
    augmented_images, augmented_labels = augment_images_with_labels_4d(
        images,
        labels,
        target_shape,
        num_augmented_per_image
    )

    logger.info(f"Normalized intensity of voxel")
    images_normalized = normalize_images_uniformly(augmented_images)

    # Data splitting
    logger.info("Splitting the dataset into train, validation, and test sets.")
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(images_normalized,augmented_labels )
    logger.debug(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    logger.debug(f"x_val shape: {x_val.shape}, y_val shape: {y_val.shape}")
    logger.debug(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

    # Model creation
    logger.info(f"Creating the model with input shape: {tuple(input_shape)}.")
    model = MyCNNModel(tuple(input_shape))
    logger.info("Model created successfully.")

    # Training
    logger.info("Starting model training.")
    model.compile_and_fit(
        x_train, y_train, x_val, y_val, x_test, y_test,
        n_epochs=args.epochs,
        batchsize=args.batchsize
    )
    logger.info("Training completed successfully.")

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
