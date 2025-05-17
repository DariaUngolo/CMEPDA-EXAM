import unittest
import tempfile
import os
import nibabel as nib
import numpy as np
from atlas_resampling import atlas_resampling  # This is the function under test


class TestAtlasResampling(unittest.TestCase):
    """
    Unit test for the `atlas_resampling()` function.

    This test verifies that a NIfTI image is correctly resampled to a new voxel size,
    specifically from 2mm isotropic to 1mm isotropic voxels.

    The test:
    - Creates a dummy NIfTI image with known voxel size and shape
    - Saves it temporarily to disk
    - Calls `atlas_resampling()` to resample the image
    - Loads the output and checks:
        1. That the new shape is correct (i.e., the volume doubles in each dimension)
        2. That the voxel dimensions are updated to the expected new spacing
    """

    def test_resampled_image_shape(self):
        """
        Test that resampling changes the image shape correctly and updates voxel spacing.

        This test creates a synthetic image with shape (5, 5, 5) and 2mm isotropic voxels.
        After resampling to 1mm isotropic, the expected shape becomes (10, 10, 10).
        """

        # Step 1: Create dummy image data with voxel values between 0 and 4
        data = np.random.randint(0, 5, size=(5, 5, 5))  # Random label-like data

        # Step 2: Define an affine matrix representing 2mm voxel spacing
        affine = np.diag([2.0, 2.0, 2.0, 1.0])  # Diagonal affine: scales XYZ by 2mm

        # Step 3: Create a NIfTI image with nibabel
        img = nib.Nifti1Image(data, affine)

        # Step 4: Create a temporary directory for input/output NIfTI files
        with tempfile.TemporaryDirectory() as tmpdirname:
            input_path = os.path.join(tmpdirname, 'input.nii.gz')
            output_path = os.path.join(tmpdirname, 'output_resampled.nii.gz')

            # Save the synthetic input image to disk
            nib.save(img, input_path)

            # Step 5: Call the resampling function to resample from 2mm to 1mm voxels
            atlas_resampling(input_path, output_path, (1.0, 1.0, 1.0))

            # Step 6: Load the output image from disk
            resampled_img = nib.load(output_path)

            # Step 7: Compute the expected new shape: each dimension should double
            expected_shape = tuple(np.array(data.shape) * 2)

            # Assert that the resampled image has the expected shape
            self.assertEqual(resampled_img.shape, expected_shape)

            # Assert that the voxel spacing (zoom) is now 1mm isotropic
            self.assertTrue(np.allclose(resampled_img.header.get_zooms()[:3], (1.0, 1.0, 1.0)))


if __name__ == '__main__':
    # Run the test suite
    unittest.main()
