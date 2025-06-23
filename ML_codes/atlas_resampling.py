import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from loguru import logger


def atlas_resampling(input_path, output_path, target_voxel_size, order=0):

    """

    Resample a NIfTI image to a specified voxel size.

    This function adjusts the spatial resolution of a NIfTI image by rescaling it to the target voxel size,
    modifying the affine transformation matrix, and saving the resulting image.

    Parameters
    ----------
    input_path : str
        Path to the input NIfTI file (.nii or .nii.gz).

    output_path : str
        Path to save the resampled NIfTI file.

    target_voxel_size : tuple of float
        Desired voxel size in millimeters (e.g., (1.0, 1.0, 1.0)).

    order : int, optional, default=0
        Interpolation order for resampling:
        - 0: Nearest neighbor (recommended for labeled atlases or segmentation masks).
        - 1: Trilinear interpolation.
        - 3: Cubic interpolation.

    Returns
    -------
    None
        The resampled NIfTI file is saved to the specified `output_path`.

    Notes
    -----
    - The function is useful for standardizing voxel dimensions or aligning anatomical and functional images.
    - Updates the affine matrix to reflect the new voxel dimensions while preserving the image origin.
    - For labeled data (e.g., atlases), use `order=0` to maintain label integrity.

    References
    ----------
    - NiBabel Documentation: https://nipy.org/nibabel/
    - SciPy Zoom Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html

    """

    # Step 1: Load the original NIfTI image

    logger.info(f"Loading NIfTI image from: {input_path}")

    img = nib.load(input_path)
    data = img.get_fdata()
    affine = img.affine
    header = img.header

    # Step 2: Get the original voxel dimensions (spacing) in mm
    original_voxel_size = header.get_zooms()[:3]
    logger.debug(f"Original voxel size: {original_voxel_size}")


    # Step 3: Check if the voxel size is already correct

    if np.allclose(original_voxel_size, target_voxel_size, atol=1e-6):
        logger.info(" Original voxel size matches target voxel size. No resampling needed.")
        # Optionally save the original image to the output path if required
        nib.save(img, output_path)
        logger.success(f"Original image saved to: {output_path}")

        return

    # Step 4: Compute scaling factors for resampling
    zoom_factors = np.array(original_voxel_size) / np.array(target_voxel_size)
    logger.info(f" Calculated zoom factors for resampling: {zoom_factors}")

    # Step 5: Perform the resampling with the chosen interpolation order
    logger.info(f" Resampling image with interpolation order {order}...")
    data_resampled = zoom(data, zoom=zoom_factors, order=order)

    # Step 6: Construct a new affine matrix with updated voxel size
    new_affine = np.eye(4)
    new_affine[:3, :3] = np.diag(target_voxel_size)
    new_affine[:3, 3] = affine[:3, 3]  # preserve image origin

    # Step 7: Create a new NIfTI image with the resampled data and updated affine
    new_img = nib.Nifti1Image(data_resampled, affine=new_affine)
    new_img.set_qform(new_affine)
    new_img.set_sform(new_affine)
    new_img.header.set_zooms(target_voxel_size)

    # Step 8: Save the new image
    nib.save(new_img, output_path)

    logger.success(f"Resampled image saved successfully to: {output_path}")


