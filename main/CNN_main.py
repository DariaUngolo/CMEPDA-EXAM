import sys
from pathlib import Path
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import nibabel as nib

from CNN_class import MyCNNModel
from utilities import ( preprocessed_images, preprocessed_images_group, split_data, augment_images_with_labels_4d, normalize_images_uniformly, adjust_image_shape )


from tensorflow.python.client import device_lib

import tensorflow as tf

from loguru import logger
# Configure loguru logger
logger.remove()  # Remove default handler
logger.add(sys.stdout, level="DEBUG", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | {level} | {message}")


# Configure GPU memory growth before TensorFlow usag
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"Memory growth enabled on {len(gpus)} GPU(s).")
    except RuntimeError as e:
        logger.error(f"Runtime error during GPU memory growth setup: {e}")

# Enable verbose logging to see where ops run (GPU/CPU)
tf.debugging.set_log_device_placement(True)

# Log detected GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"[INFO] {len(gpus)} GPU(s) detected:")
    for gpu in gpus:
        print(f"    -> {gpu}")
else:
    logger.warning("No GPU detected. Training will use CPU.")



def ask_yes_no_prompt(prompt_message, default="N"):
    """

    Prompt the user with a yes/no question and return their response as a boolean.

    Parameters
    ----------

    prompt_message : str
        The message or question to display to the user.
    default : str, optional, default='N'
        The default response if the user provides no input.
        Must be either 'Y' (yes) or 'N' (no).

    Returns
    -------

    bool
        True if the user selects 'Y' (yes), False if the user selects 'N' (no).

    Notes
    -----

    The user can input 'Y'/'y' for yes or 'N'/'n' for no.
    If no input is given, the default value is used.
    Prompts repeatedly until a valid response is provided.

    """

    while True:
        # Determine the displayed default option
        default_display = "Y" if default.upper() == "Y" else "N"
        # Ask the user for input
        user_input = input(f"{prompt_message} [Y/N] (default: {default_display}): ").strip().lower()

        # Use default if no input is provided
        if not user_input:
            user_input = default.lower()

        # Return True for 'y' and False for 'n'
        if user_input in ["y", "n"]:
            return user_input == "y"

        # Prompt again for invalid input
        logger.warning("Invalid input. Please enter 'Y' or 'N'.")


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
        '--epochs', type=int, default=100,
        help="Number of training epochs (default: %(default)s)."
    )
    parser.add_argument(
        '--batchsize', type=int, default=20,
        help="Batch size for training (default: %(default)s)."
    )

    #trained model
    parser.add_argument(
        "--use_trained_model", action="store_true",
        help="Skip training; classify images with a saved model."
    )
    parser.add_argument(
        "--trained_model_path", type=str,
        help="Path to saved model (joblib). Required if --use_trained_model is set."
    )
    parser.add_argument(
        "--nifti_image_path", type=str,
        help="Image path "
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
    logger.info("Available devices:")
    logger.info(device_lib.list_local_devices())

    if tf.config.list_physical_devices('GPU'):
        logger.info("TensorFlow is using GPU.")
    else:
        logger.warning("TensorFlow is not using GPU, running on CPU.")


    roi_ids=(165,166)

    # Classification only mode
    if args.use_trained_model:
        logger.info("Using pre-trained model for classification.")

        if not args.trained_model_path:
            logger.error("You must specify --trained_model_path if using --use_trained_model")
            sys.exit(1)
        if not args.nifti_image_path:
            logger.error("You must specify --nifti_image_path to classify new images")
            sys.exit(1)

        # Load model within strategy scope if needed (e.g. multi-GPU)
        with strategy.scope():
            trained_model = tf.keras.models.load_model(args.trained_model_path)

        trained_model = tf.keras.models.load_model(args.trained_model_path)
        model_shape = (121,145,29,1)



        # Load and preprocess the image
        nifti_img_preprocessed = preprocessed_images(args.nifti_image_path, args.atlas_path, tuple(roi_ids))
        nifti_img_preprocessed = adjust_image_shape(nifti_img_preprocessed,  model_shape )
        images_normalized = normalize_images_uniformly(nifti_img_preprocessed,)


        prediction = trained_model.predict(images_normalized)
        logger.info(f"Predicted probability for class 1: {prediction[0][0]}")

        return


    num_augmented_per_image = 3


    # Preprocessing images and labels
    logger.info("Starting image preprocessing.")
    images, labels = preprocessed_images_group(args.image_folder, args.atlas_path, args.metadata, tuple(roi_ids))
    logger.debug(f"Preprocessed image dimensions: {images.shape}")
    logger.debug(f"Preprocessed label dimensions: {labels.shape}")



    target_shape = images.shape[1:4]
    input_shape = images.shape[1:]

    logger.info(f"Using target shape for augmentation derived from training data: {target_shape}")
    logger.info(f"Using input shape for the CNN model: {input_shape}")

    # Normalize intensities before augmentation
    logger.info(f"Normalized intensity of voxel")
    images_normalized = normalize_images_uniformly(images)

    # Split dataset into train/val/test
    logger.info("Splitting the dataset into train, validation, and test sets (70-15-15).")
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(images_normalized, labels )

    logger.debug(f"x_train shape before data-augumentation: {x_train.shape}, y_train shape: {y_train.shape}")
    logger.debug(f"x_val shape: {x_val.shape}, y_val shape: {y_val.shape}")
    logger.debug(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

    # Perform data augmentation
    logger.info("Starting image augmentation on training data only.")
    x_train_augmented, y_train_augmented = augment_images_with_labels_4d(
        x_train,
        y_train,
        target_shape,
        num_augmented_per_image
     )

    logger.debug(f"After data-augumentation... x_train shape: {x_train_augmented.shape}, y_train shape: {y_train_augmented.shape}")

    # Model creation
    logger.info(f"Creating the model with input shape: {tuple(input_shape)}.")
    model = MyCNNModel(input_shape=input_shape)




    # Compile and train the model
    logger.info("Starting model training.")
    model.compile_and_fit(
            x_train_augmented, y_train_augmented, x_val, y_val, x_test, y_test,
            n_epochs=args.epochs,
            batchsize=args.batchsize
        )
    logger.success("Training completed successfully.")

    #model.load_model("model_full.h5")

    if model is not None:
        do_classify = ask_yes_no_prompt("Do you want to classify new images now? Y/N", default="N")
        if do_classify:
            model_shape = model.input_shape

            # Load NIfTI image
            nifti_img_preprocessed = preprocessed_images(args.nifti_image_path, args.atlas_path, tuple(roi_ids))

            nifti_img_preprocessed = adjust_image_shape(nifti_img_preprocessed,  model_shape )
            images_normalized = normalize_images_uniformly(nifti_img_preprocessed,)

            # Perform prediction.
            prediction = model.predict(images_normalized)
            logger.info(f"Predicted probability for class 1: {prediction[0][0]}")



if __name__ == '__main__':
    args = parse_arguments()
    main(args)