import sys
from pathlib import Path
import os
import argparse
import logging

from CNN_class import MyCNNModel
from utilities import preprocessed_images, preprocessed_images_group, split_data, augment_images_with_labels_4d, normalize_images_uniformly, adjust_image_shape
import nibabel as nib


# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


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
        print("Invalid input. Please enter 'Y' or 'N'.")


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
        '--epochs', type=int, default=400,
        help="Number of training epochs (default: %(default)s)."
    )
    parser.add_argument(
        '--batchsize', type=int, default=50,
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
    roi_ids=(165,166)

    # === Step 0: Classification only mode ===
    if args.use_trained_model:

        if not args.trained_model_path:
            logger.error("Devi specificare --trained_model_path se usi --use_trained_model")
            sys.exit(1)
        if not args.nifti_image_path:
            logger.error("Devi specificare --nifti_image_path per classificare nuove immagini")
            sys.exit(1)

        trained_model = MyCNNModel()    
        trained_model.load_model(args.trained_model_path)
        model_shape = trained_model.input_shape 

            
        # Load NIfTI image
        nifti_img_preprocessed = preprocessed_images(args.nifti_image_path, args.atlas_path, tuple(roi_ids)) 
        # Adatta dimensioni all'input shape
        nifti_img_preprocessed = adjust_image_shape(nifti_img_preprocessed,  model_shape )
        images_normalized = normalize_images_uniformly(nifti_img_preprocessed,)

        # Perform prediction. Fai predizione
        prediction = trained_model.predict(images_normalized)
        print(f"[RESULT] Predicted probability for class 1: {prediction[0][0]}")

        return

    
    num_augmented_per_image = 2


    # Image preprocessing
    logger.info("Starting image preprocessing.")
    images, labels = preprocessed_images_group(args.image_folder, args.atlas_path, args.metadata, tuple(roi_ids))
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

    model.load_model("model_full.h5")

    if model is not None:
        do_classify = ask_yes_no_prompt("Do you want to classify new images now? Y/N", default="N")
        if do_classify:
            model_shape = model.input_shape 

            # Load NIfTI image
            nifti_img_preprocessed = preprocessed_images(args.nifti_image_path, args.atlas_path, tuple(roi_ids)) 
            # Adatta dimensioni all'input shape
            nifti_img_preprocessed = adjust_image_shape(nifti_img_preprocessed,  model_shape )
            images_normalized = normalize_images_uniformly(nifti_img_preprocessed,)

            # Perform prediction. Fai predizione
            prediction = model.predict(images_normalized)
            print(f"[RESULT] Predicted probability for class 1: {prediction[0][0]}")



if __name__ == '__main__':
    args = parse_arguments()
    main(args)