import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))


import argparse
from pathlib import Path
import matlab.engine
from loguru import logger
import joblib
from tabulate import tabulate


from ML_codes.feature_extractor import feature_extractor
from ML_codes.classifiers import (
    RFPipeline_noPCA,
    RFPipeline_PCA,
    RFPipeline_RFECV,
    SVM_simple
)
from ML_codes.atlas_resampling import atlas_resampling
from ML_codes.apply_trained_model import feature_extractor_independent_dataset, classify_independent_dataset

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



def parse_arguments():
    """
    Parses command-line arguments for the classification pipeline.
    Returns
    -------
    argparse.Namespace
        Parsed arguments for resampling, feature extraction, and classification.
    """
    parser = argparse.ArgumentParser(
        description="""
         Brain MRI classification pipeline using region-based feature extraction.
        The pipeline leverages MATLAB for extracting features from NIfTI images
        based on a parcellation atlas, and uses machine learning models in Python
        (Random Forest, SVM) to classify between Alzheimer (AD) and control (CN) subjects.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # === Input paths ===
    parser.add_argument(
        "--folder_path", required= False, type=str,
        help="""Path to the folder containing subject NIfTI images to be analyzed.
        Each file should represent one subject. Supported formats: .nii, .nii.gz"""
    )

    parser.add_argument(
        "--atlas_file", required=True, type=str,
        help="""Path to the original brain atlas NIfTI file. Each voxel contains
        a numeric label indicating the region-of-interest (ROI)."""
    )

    parser.add_argument(
        "--atlas_file_resized", required=True, type=str,
        help="""Path where the resampled atlas NIfTI file will be saved.
        The atlas will be resampled to match the voxel resolution of the input images."""
    )

    parser.add_argument(
        "--atlas_txt", required=True, type=str,
        help="""Text file containing the ROI labels corresponding to the atlas.
        Each row should include an index and the corresponding region name."""
    )

    parser.add_argument(
        "--metadata_csv",required= False, type=str,
        help="""TSV file containing subject metadata and diagnostic labels.
        It must include: subject ID and diagnosis label (e.g., 'Normal', 'AD')."""
    )


    parser.add_argument(
        "--matlab_path", required=True, type=str,
        help="""Path to the folder containing the MATLAB script `feature_extractor.m`.
        This path will be added to the MATLAB environment."""
    )

    # === Classifier configuration ===
    parser.add_argument(
        "--classifier",
        choices=["rf", "svm"],
        help="""Classifier to use:
        - 'rf': Random Forest .
        - 'svm': Support Vector Machine (linear or RBF kernel)."""
    )

    parser.add_argument(
        "--n_iter", type=int, default=10,
        help="Number of parameter combinations to sample in RandomizedSearchCV for Random Forest."
    )

    parser.add_argument(
        "--cv", type=int, default=5,
        help="Number of cross-validation folds to use."
    )

    parser.add_argument(
        "--kernel", choices=["linear", "rbf"], default="rbf",
        help="Kernel type to use for SVM (only applicable if classifier is 'svm')."
    )

    # === Inference option: use pre-trained model for classification mode ===
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
        help="Comma-separated NIfTI image path(s) to classify when using --use_trained_model."
    )

    return parser.parse_args()

    # === Argument validation ===
    if args.use_trained_model:
        missing = []
        if not args.trained_model_path:
            missing.append("--trained_model_path")
        if not args.nifti_image_path:
            missing.append("--nifti_image_path")
        if not args.atlas_file_resized:
            missing.append("--atlas_file_resized")
        if not args.atlas_txt:
            missing.append("--atlas_txt")
        if not args.matlab_path:
            missing.append("--matlab_path")
        if missing:
            parser.error(f"The following arguments are required when using --use_trained_model: {', '.join(missing)}")
    else:
        # Training mode: check all training-related arguments
        required = {
            "--folder_path": args.folder_path,
            "--atlas_file": args.atlas_file,
            "--atlas_file_resized": args.atlas_file_resized,
            "--atlas_txt": args.atlas_txt,
            "--metadata_csv": args.metadata_csv,
            "--matlab_path": args.matlab_path,
            "--classifier": args.classifier,
        }
        missing = [k for k, v in required.items() if v is None]
        if missing:
            parser.error(f"The following arguments are required for training: {', '.join(missing)}")

    return args


def main():
    """
    Brain MRI classification pipeline using region-based features.

    This function runs the full pipeline for brain MRI classification with two modes: training and inference.

    Parameters
    ----------
    --train : bool, optional
        If specified, runs the pipeline in training mode.
    --inference : bool, optional
        If specified, runs the pipeline in inference mode.
    --subjects_dir : str
        Path to the directory containing subject NIfTI images.
    --atlas_path : str
        Path to the brain atlas NIfTI file.
    --clinical_data : str, optional
        Path to the CSV file with clinical labels or metadata. Required in training mode.
    --model_path : str
        Path to save the trained model (training mode) or load the model (inference mode).
    --classifier : {'rf', 'svm'}, optional
        Classifier type: "rf" (Random Forest) or "svm" (Support Vector Machine). Default is "rf".
    --resample : bool, optional
        If set, the atlas will be resampled to each subject's MRI resolution.
    --matlab_function : str, optional
        MATLAB function name for feature extraction. Defaults to a predefined function.

    Returns
    -------
    None
        In training mode, saves the trained classifier model to disk.
        In inference mode, prints classification results and probabilities in a tabular format.

    Raises
    ------
    FileNotFoundError
        If any specified file or directory path does not exist.
    ValueError
        If incompatible or missing arguments are provided (e.g., missing clinical_data in training).

    Notes
    -----
    - MATLAB and MATLAB Engine API for Python must be installed and configured.
    - All subject NIfTI images must be preprocessed and compatible with the specified atlas.
    - For inference, the model_path must point to a valid `.joblib` file.
    - Feature extraction relies on a MATLAB pipeline invoked from Python.

    Examples
    --------
    Training:

    >>> python main.py --train --subjects_dir ./data/ --atlas_path ./atlas.nii.gz --clinical_data ./labels.csv --model_path ./model.joblib

    Inference:

    >>> python main.py --inference --subjects_dir ./new_subjects/ --atlas_path ./atlas.nii.gz --model_path ./model.joblib

    """



    args = parse_arguments()


    # === Step 0: Inference only mode ===

    if args.use_trained_model:
        logger.info("Using pre-trained model for classification.")

        nifti_paths = [p.strip() for p in args.nifti_image_path.split(",") if p.strip()]
        results = []

        for image_path in nifti_paths:
            logger.info(f"Extracting features from: {image_path}")
            df_mean, df_std, df_volume, df_mean_std, df_mean_volume, df_std_volume, df_mean_std_volume = feature_extractor_independent_dataset(
                image_path,
                args.atlas_file_resized,
                args.atlas_txt,
                args.matlab_path
            )
            classification, probability = classify_independent_dataset(df_mean_std, args.trained_model_path)

            if classification == 1:
                    selected_probability = probability[1]
            else:
                    selected_probability = probability[0]

            results.append({
                "Image": os.path.basename(image_path),
                "Prediction": classification,

                "Probability": f"{selected_probability:.2f}"

            })

        # Display results in a table
        print("\nClassification Results:")
        print(tabulate(results, headers="keys", tablefmt="fancy_grid"))

        logger.success("Classification completed.")
        return

    # === Step 1: Resample the atlas ===
    target_voxel = (1.5, 1.5, 1.5)
    logger.info(f" Resampling atlas to voxel size {target_voxel}...")
    atlas_resampling(args.atlas_file, args.atlas_file_resized, target_voxel, order=0)

    # === Step 2: Start MATLAB engine for feature extraction ===
    logger.info("🔧 Starting MATLAB engine...")
    eng = matlab.engine.start_matlab()
    eng.addpath(args.matlab_path, nargout=0)

    logger.info(" Extracting features from NIfTI images using MATLAB...")
    df_mean, df_std, df_volume, df_mean_std, df_mean_volume, df_std_volume, df_mean_std_volume, diagnostic_group_labels = feature_extractor(
        args.folder_path,
        args.atlas_file_resized,
        args.atlas_txt,
        args.metadata_csv,
        args.matlab_path
    )

    eng.quit()
    logger.success(" Feature extraction completed successfully.")

    # === Step 3: Sanity check ===
    if df_mean.shape[0] != diagnostic_group_labels.shape[0]:
        logger.error(" Mismatch in number of subjects between features and metadata.")
        return

    # === Step 4: Classification ===
    logger.info(f" Running classifier: {args.classifier}")

    model = None  # To store the trained model for later saving

    if args.classifier == "rf":
        use_pca = ask_yes_no_prompt("Principal component analysis (PCA)? Y or N:", default="N")

        if use_pca:
            logger.info(" Applying Random Forest with PCA...")
            model = RFPipeline_PCA(df_mean_std, diagnostic_group_labels, args.n_iter, args.cv)

        else:
            use_rfe = ask_yes_no_prompt("Recursive Feature Elimination (RFE)? Y or N:", default="N")

            if use_rfe:
                logger.info(" Applying Random Forest with RFECV...")
                model = RFPipeline_RFECV(df_mean_std, diagnostic_group_labels, args.n_iter, args.cv)
            else:
                logger.info(" Applying Random Forest without PCA or RFE...")
                model = RFPipeline_noPCA(df_mean_std, diagnostic_group_labels, args.n_iter, args.cv)

    elif args.classifier == "svm":
        logger.info(" Applying Support Vector Machine...")
        model = SVM_simple(df_mean_std, diagnostic_group_labels, ker=args.kernel)

    # === Step 5: Save the trained model ===

    if model is not None:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        model_dir = os.path.join(base_dir, "trained_model")  # o "trained_model" se la cartella è minuscola

        if args.classifier == "rf":
            if use_pca:
                model_filename = os.path.join(model_dir, "trained_model_rf_pca.joblib")
            elif use_rfe:
                model_filename = os.path.join(model_dir, "trained_model_rf_rfecv.joblib")
            else:
                model_filename = os.path.join(model_dir, "trained_model_rf.joblib")
        elif args.classifier == "svm":
            model_filename = os.path.join(model_dir, f"trained_model_svm_{args.kernel}.joblib")
        else:
            model_filename = os.path.join(model_dir, "trained_model.joblib")

    joblib.dump(model, model_filename)
    logger.success(f" Trained model saved to '{model_filename}'")

    # === Step 6: Classify the independent dataset ===
    # Ask user if they want to classify new images now
    if model is not None:
        do_classify = ask_yes_no_prompt("Do you want to classify new images now? Y/N", default="N")
        if do_classify:
            results = []
            while True:
                image_path = prompt_for_valid_folder_path()
                if image_path is None:
                    logger.info("Classification aborted by user.")
                    break

                logger.info(f"Extracting features for image '{image_path}'...")
                # Extract features for this image using the independent function
                df_mean, df_std, df_volume, df_mean_std, df_mean_volume, df_std_volume, df_mean_std_volume = feature_extractor_independent_dataset(
                    image_path,
                    args.atlas_file_resized,
                    args.atlas_txt,
                    args.matlab_path
                )

                # We classify and get the classification and probability, then display the results
                classification, probability = classify_independent_dataset(df_mean_std, model_filename)
                #
                if classification == 1:
                    selected_probability = probability[1]
                else:
                    selected_probability = probability[0]

                results.append({
                "Image": os.path.basename(image_path),
                "Prediction": classification,
                "Probability": f"{selected_probability:.2f}"
                })

                # Display results in a table
                print("\nClassification Results:")
                print(tabulate(results, headers="keys", tablefmt="fancy_grid"))

                continue_classify = ask_yes_no_prompt("Classify another image? Y/N", default="N")
                if not continue_classify:
                    break

    logger.success(" Classification pipeline completed successfully.")


if __name__ == "__main__":
    main()

