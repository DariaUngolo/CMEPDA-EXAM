import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))


import argparse
from pathlib import Path
import matlab.engine
from loguru import logger

from ML_codes.feature_extractor import feature_extractor
from ML_codes.classifiers_unified import (
    RFPipeline_noPCA,
    RFPipeline_PCA,
    RFPipeline_RFECV,
    SVM_simple
)
from ML_codes.atlas_resampling import atlas_resampling

def ask_yes_no_prompt(prompt_message, default="N"):
    """
    Prompt the user with a yes/no question and return their response as a boolean.

    **Parameters**:
    - **prompt_message** (`str`):
        The message or question to display to the user.
    - **default** (`str`, optional, default=`"N"`):
        The default response if the user provides no input.
        Must be either `"Y"` (yes) or `"N"` (no).

    **Returns**:
    - `bool`:
        `True` if the user selects `"Y"` (yes), `False` if the user selects `"N"` (no).

    **Notes**:
    - The user can input `"Y"`/`"y"` for yes or `"N"`/`"n"` for no.
    - If no input is given, the default value is used.
    - Prompts repeatedly until a valid response is provided.
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
    Parses command-line arguments for the classification pipeline.
    Returns
    -------
    argparse.Namespace
        Parsed arguments for resampling, feature extraction, and classification.
    """
    parser = argparse.ArgumentParser(
        description="""
        🧠 Brain MRI classification pipeline using region-based feature extraction.
        The pipeline leverages MATLAB for extracting features from NIfTI images
        based on a parcellation atlas, and uses machine learning models in Python
        (Random Forest, SVM) to classify between Alzheimer (AD) and control (CN) subjects.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # === Input paths ===
    parser.add_argument(
        "--folder_path", required=True, type=str,
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
        "--metadata_csv", required=True, type=str,
        help="""TSV file containing subject metadata and diagnostic labels.
        It must include: subject ID and diagnosis label (e.g., 'Normal', 'AD')."""
    )

    parser.add_argument(
        "--output_prefix", required=True, type=str,
        help="""Prefix to use for intermediate output CSV files produced by the MATLAB feature extraction step."""
    )

    parser.add_argument(
        "--matlab_path", required=True, type=str,
        help="""Path to the folder containing the MATLAB script `feature_extractor.m`.
        This path will be added to the MATLAB environment."""
    )

    # === Classifier configuration ===
    parser.add_argument(
        "--classifier", required=True,
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

    return parser.parse_args()


def main():
    """
    Run the complete brain MRI classification pipeline.

    This function orchestrates the entire workflow:
    1. Resampling the brain atlas to match MRI resolution.
    2. Extracting regional features using MATLAB.
    3. Training and evaluating the selected classification model.

    **Notes**:
    - User inputs and parameters are parsed from command-line arguments.
    - Supports Random Forest with optional PCA or RFECV, and SVM classifiers.
    - Logs progress and handles basic sanity checks.
    """
    args = parse_arguments()

    # === Step 1: Resample the atlas ===
    target_voxel = (1.5, 1.5, 1.5)
    logger.info(f"📏 Resampling atlas to voxel size {target_voxel}...")
    atlas_resampling(args.atlas_file, args.atlas_file_resized, target_voxel, order=0)

    # === Step 2: Start MATLAB engine for feature extraction ===
    logger.info("🔧 Starting MATLAB engine...")
    eng = matlab.engine.start_matlab()
    eng.addpath(args.matlab_path, nargout=0)

    logger.info("📊 Extracting features from NIfTI images using MATLAB...")
    df_mean, df_std, df_volume, df_mean_std, df_mean_volume, df_std_volume, df_mean_std_volume, diagnostic_group_labels = feature_extractor(
        args.folder_path,
        args.atlas_file_resized,
        args.atlas_txt,
        args.metadata_csv,
        args.output_prefix,
        args.matlab_path
    )

    eng.quit()
    logger.success("✅ Feature extraction completed successfully.")

    # === Step 3: Sanity check ===
    if df_mean.shape[0] != diagnostic_group_labels.shape[0]:
        logger.error("❌ Mismatch in number of subjects between features and metadata.")
        return

    # === Step 4: Classification ===
    logger.info(f"🚀 Running classifier: {args.classifier}")


    # Check which classifier is selected by the user
    if args.classifier == "rf":
        # Ask user if they want to apply PCA before Random Forest classification
        use_pca = ask_yes_no_prompt("Principal component analysis (PCA)? Y or N:", default="N")

        # If PCA is chosen, run the Random Forest pipeline with PCA
        if use_pca:
            logger.info(" Applying Random Forest with PCA...")
            RFPipeline_PCA(df_mean_std, diagnostic_group_labels, args.n_iter, args.cv)

        else:
            # If PCA is not chosen, ask if Recursive Feature Elimination (RFE) should be applied
            use_rfe = ask_yes_no_prompt("Recursive Feature Elimination (RFE)? Y or N:", default="N")

            if use_rfe:
                # Run the Random Forest pipeline with RFE and cross-validation (RFECV)
                logger.info(" Applying Random Forest with RFECV...")
                RFPipeline_RFECV(df_mean_std, diagnostic_group_labels, args.n_iter, args.cv)
            else:
                # Run the standard Random Forest pipeline without PCA or RFE
                logger.info(" Applying Random Forest without PCA or RFE...")
                RFPipeline_noPCA(df_mean_std, diagnostic_group_labels, args.n_iter, args.cv)

    # If SVM is selected as classifier
    elif args.classifier == "svm":
        # Log the choice and run the SVM pipeline with specified kernel
        logger.info(" Applying Support Vector Machine...")
        SVM_simple(df_mean_std, diagnostic_group_labels, ker=args.kernel)




    logger.success("🎯 Classification pipeline completed successfully.")

    



if __name__ == "__main__":
    main()

