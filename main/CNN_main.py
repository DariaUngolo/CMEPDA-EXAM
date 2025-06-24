import sys
from pathlib import Path
import os


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import nibabel as nib

from CNN_codes.CNN_class import MyCNNModel
from CNN_codes.utilities import ( preprocessed_images, preprocessed_images_group, split_data, augment_images_with_labels_4d, normalize_images_uniformly, adjust_image_shape )

import numpy
from tensorflow.python.client import device_lib

import tensorflow as tf

seed = 4
random.seed(seed)
numpy.random.seed(seed)
tf.random.set_seed(seed)

from loguru import logger
# Configure loguru logger
logger.remove()  # Remove default handler
logger.add(sys.stdout, level="DEBUG", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | {level} | {message}")
from tabulate import tabulate

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

def prompt_for_valid_folder_path():
    """
    Prompt the user to input a valid folder path containing NIfTI images for classification.
    Keeps asking until a valid folder path is given or the user chooses to quit.

    Returns
    -------
    str or None
        Valid path to the folder or None if the user wants to quit.
    """
    while True:
        folder_path = input("Please enter the path to the folder containing NIfTI images to classify (or type 'quit' to exit): ").strip()
        if folder_path.lower() == "quit":
            return None
        if os.path.isdir(folder_path):
            return folder_path
        else:
            print(f"Error: The folder '{folder_path}' does not exist or is not accessible. Please try again.")



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
        '--image_folder', type=str,
        help="Path to the folder containing NIfTI images."
    )
    parser.add_argument(
        '--atlas_path', type=str, required=True,
        help="Path to the NIfTI atlas file."
    )
    parser.add_argument(
        '--metadata', type=str,
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
        help="Path to saved model (.h5). Required if --use_trained_model is set."
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

        trained_model = tf.keras.models.load_model(args.trained_model_path)
        #model_shape = (121,145,47,1)
        model_shape = trained_model .input_shape
        target_shape = trained_model.input_shape[1:4]
        logger.info(f"Target shape: {target_shape}")



        # Load and preprocess the image
        nifti_img_preprocessed = preprocessed_images(args.nifti_image_path, args.atlas_path, tuple(roi_ids))
        nifti_img_preprocessed = adjust_image_shape(nifti_img_preprocessed,  target_shape )
        images_normalized = normalize_images_uniformly(nifti_img_preprocessed,)


        prediction = trained_model.predict(images_normalized)
        threshold = 0.5
        classification = 1 if prediction[0][0] >= threshold else 0


        if classification == 1:
            selected_probability = prediction[0][0]
        else:
            selected_probability = 1- prediction[0][0]

        results.append({
        "Image": os.path.basename(nifti_image_path),
        "Prediction": classification,
        "Probability": f"{selected_probability:.2f}"
        })

        # Display results in a table
        print("\nClassification Results:")
        print(tabulate(results, headers="keys", tablefmt="fancy_grid"))

        return




    # Preprocessing images and labels
    images, labels = preprocessed_images_group(args.image_folder, args.atlas_path, args.metadata, tuple(roi_ids))

    target_shape = images.shape[1:4]
    input_shape = images.shape[1:]

    # Normalize intensities before augmentation
    images_normalized = normalize_images_uniformly(images)

    # Split dataset into train/val/test
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(images_normalized, labels )


    # Perform data augmentation
    logger.info("Starting image augmentation on training data only.")
    x_train_augmented, y_train_augmented = augment_images_with_labels_4d(
        x_train,
        y_train,
        target_shape
    )

    # Model creation
    model = MyCNNModel(input_shape=input_shape)




    # Compile and train the model
    model.compile_and_fit(
            x_train_augmented, y_train_augmented, x_val, y_val, x_test, y_test,
            n_epochs=args.epochs,
            batchsize=args.batchsize
        )
    logger.success("Training completed successfully.")

    model.save_model("trained_model.h5")

    if model is not None:
        do_classify = ask_yes_no_prompt("Do you want to classify new images now? Y/N", default="N")
        if do_classify:
            results=[]
            while True:
                nifti_image_path = prompt_for_valid_folder_path()
                if nifti_image_path is None:
                    logger.info("Classification aborted by user.")
                    break



                model_shape = input_shape

                # Load NIfTI image
                nifti_img_preprocessed = preprocessed_images(nifti_image_path, args.atlas_path, tuple(roi_ids))

                nifti_img_preprocessed = adjust_image_shape(nifti_img_preprocessed,  model_shape )
                images_normalized = normalize_images_uniformly(nifti_img_preprocessed)

                # Perform prediction.
                prediction = model.predict(images_normalized)
                logger.info(f"Predicted probability for class 1: {prediction[0][0]}")

                threshold = 0.5
                classification = 1 if prediction[0][0] >= threshold else 0


                if classification == 1:
                    selected_probability = prediction[0][0]
                else:
                    selected_probability = 1- prediction[0][0]

                results.append({
                "Image": os.path.basename(nifti_image_path),
                "Prediction": classification,
                "Probability": f"{selected_probability:.2f}"
                })

                # Display results in a table
                print("\nClassification Results:")
                print(tabulate(results, headers="keys", tablefmt="fancy_grid"))

                continue_classify = ask_yes_no_prompt("Classify another image? Y/N", default="N")
                if not continue_classify:
                    break



if __name__ == '__main__':
    args = parse_arguments()
    main(args)