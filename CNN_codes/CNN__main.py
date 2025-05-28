import sys
import os
import argparse
from pathlib import Path
from loguru import logger

from CNN_class import MyCNNModel
from utilities import preprocess_nifti_images, split_data, augment_images_with_labels_4d


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
        '--epochs', type=int, default=700,
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

    num_augmented_per_image = 2    # Number of augmentations per image
    roi_ids= (165,166)

    logger.debug(f"Received command-line arguments: {args}")

    # Preprocess images
    logger.info("Starting preprocessing of NIfTI images...")
    images, labels = preprocess_nifti_images(args.image_folder, args.atlas_path, args.metadata, tuple(roi_ids))
    logger.debug(f"Images shape: {images.shape}")
    logger.debug(f"Labels shape: {labels.shape}")

    target_shape = images.shape[1:4]
    input_shape = images.shape[1:]
    logger.info(f"Using target shape for augmentation derived from training data: {target_shape}")

    # Data augmentation
    logger.info("Performing data augmentation on images...")
    augmented_images, augmented_labels = augment_images_with_labels_4d(
        images,
        labels,
        target_shape,
        num_augmented_per_image
    )

    # Split data into train, validation, and test sets
    logger.info("Splitting dataset into training, validation, and test sets...")
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(augmented_images, augmented_labels)
    logger.debug(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    logger.debug(f"x_val shape: {x_val.shape}, y_val shape: {y_val.shape}")
    logger.debug(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

    # Create CNN model
    logger.info(f"Creating CNN model with input shape: {tuple(input_shape)}")
    model = MyCNNModel(tuple(input_shape))
    logger.info("Model successfully created.")

    # Train the model
    logger.info("Starting model training...")
    model.compile_and_fit(
        x_train, y_train,
        x_val, y_val,
        x_test, y_test,
        n_epochs=args.epochs,
        batchsize=args.batchsize
    )
    logger.success("Training completed successfully.")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
