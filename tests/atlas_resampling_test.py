import unittest
import tempfile
import os
import nibabel as nib
import numpy as np
from atlas_resampling import atlas_resampling  # Function under test
from loguru import logger

class TestAtlasResampling(unittest.TestCase):
    """
    Unit test for the `atlas_resampling()` function.

    This test verifies that a NIfTI image is properly resampled to a specified voxel size,
    specifically testing the upsampling from 2mm isotropic voxels to 1mm isotropic voxels.

    Test steps:
    - Create a synthetic NIfTI image with known shape and 2mm voxel size.
    - Save the image to a temporary file.
    - Use `atlas_resampling()` to resample the image to 1mm isotropic voxels.
    - Load the resampled image and verify:
        * The shape is doubled in each dimension (due to halving voxel size).
        * The voxel dimensions are correctly updated to (1.0, 1.0, 1.0).
    """

    def test_resampled_image_shape(self):
        """
        Test that a NIfTI image with 2mm voxel spacing is correctly resampled to 1mm spacing.

        Specifically:
        - Input image shape: (5, 5, 5)
        - Expected output shape: (10, 10, 10) after upsampling
        - Confirm that voxel size changes from (2.0, 2.0, 2.0) to (1.0, 1.0, 1.0)
        """
        logger.info("Creating synthetic NIfTI image with shape (5, 5, 5) and voxel size 2mm isotropic.")
        data = np.random.randint(0, 5, size=(5, 5, 5))

        # Define affine with 2mm isotropic voxel size
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        img = nib.Nifti1Image(data, affine)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, 'input.nii.gz')
            output_path = os.path.join(tmpdir, 'output_resampled.nii.gz')

            logger.debug(f"Saving synthetic image to temporary file: {input_path}")
            nib.save(img, input_path)

            logger.info("Calling atlas_resampling to resample image to 1mm isotropic voxels.")
            atlas_resampling(input_path, output_path, (1.0, 1.0, 1.0))

            logger.debug(f"Loading resampled image from {output_path}")
            resampled_img = nib.load(output_path)

            expected_shape = tuple(np.array(data.shape) * 2)
            logger.info(f"Expected shape after resampling: {expected_shape}")
            logger.info(f"Actual shape of resampled image: {resampled_img.shape}")
            self.assertEqual(resampled_img.shape, expected_shape, 
                             msg=f"Expected shape {expected_shape}, got {resampled_img.shape}")

            new_voxel_size = resampled_img.header.get_zooms()[:3]
            logger.info(f"Voxel size after resampling: {new_voxel_size}")
            self.assertTrue(np.allclose(new_voxel_size, (1.0, 1.0, 1.0)),
                            msg=f"Expected voxel size (1.0, 1.0, 1.0), got {new_voxel_size}")


if __name__ == '__main__':
    logger.info("Starting unittest for atlas resampling...")
    unittest.main()
