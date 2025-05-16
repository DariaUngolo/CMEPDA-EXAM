import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

def resample_nifti(input_path, output_path, target_voxel_size, order=0):
    """
    Resamples a NIfTI image to the desired voxel size.

    Parameters:
    - input_path: str, path to the original NIfTI file.
    - output_path: str, path to save the resampled NIfTI file.
    - target_voxel_size: tuple (sx, sy, sz) in mm, the desired voxel dimensions.
    - order: int, interpolation order (0=nearest neighbor, 1=linear, 3=cubic). Use 0 for labeled atlases.
    """

    # Step 1: Load the original NIfTI file.
    img = nib.load(input_path)  # Load the NIfTI file.
    data = img.get_fdata()  # Extract the image data as a NumPy array.
    header = img.header  # Get the header containing metadata about the image.
    affine = img.affine  # Get the affine transformation matrix.

    # Step 2: Extract the original voxel dimensions (in mm).
    original_voxel_size = header.get_zooms()[:3]  # Get voxel size along x, y, z axes.

    # Step 3: Compute the scaling factors needed to match the target voxel size.
    zoom_factors = np.array(original_voxel_size) / np.array(target_voxel_size)
    print(f"Zoom factors: {zoom_factors}")  # Debug information to check scaling.

    # Step 4: Resample the image data using the computed zoom factors.
    # Note: The 'order' parameter controls the type of interpolation.
    # Use order=0 for labeled data (e.g., brain region atlases) to avoid introducing fractional labels.
    data_resampled = zoom(data, zoom=zoom_factors, order=order)

    # Step 5: Update the affine transformation matrix to reflect the new voxel size.
    # Affine[:3, :3] stores the scaling and rotation information.
    new_affine = affine.copy()  # Copy the original affine matrix.
    new_affine[:3, :3] = affine[:3, :3] / np.diag(affine[:3, :3])[:, None] * target_voxel_size

    # Step 6: Create a new NIfTI image with the resampled data and updated affine matrix.
    new_img = nib.Nifti1Image(data_resampled, affine=new_affine)

    # Update the header to include the new voxel size.
    new_img.header.set_zooms(target_voxel_size)

    # Step 7: Save the resampled NIfTI image to the specified output path.
    nib.save(new_img, output_path)
    print(f"Resampled NIfTI image saved to: {output_path}")

if __name__ == "__main__":
    # Input and output file paths for the NIfTI file.
    input_atlas = "C:\\Users\\daria\\OneDrive\\Desktop\\ESAME\\BN_Atlas_246_2mm.nii.gz"  # Path to the original file.
    output_atlas = "C:\\Users\\daria\\OneDrive\\Desktop\\BN_Atlas_246_1.5mm.nii.gz"  # Path for the output file.

    # Desired voxel size (in millimeters).
    target_voxel = (1.5, 1.5, 1.5)

    # Use nearest-neighbor interpolation (order=0) as it is ideal for labeled atlases.
    resample_nifti(input_atlas, output_atlas, target_voxel, order=0)