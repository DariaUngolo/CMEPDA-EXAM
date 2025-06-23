import unittest
import numpy as np
import tensorflow as tf
import pandas as pd
from unittest import mock
from loguru import logger
from CNN_codes.utilities import (
    pad_images,
    random_rotate_tf,
    random_zoom_and_crop_tf,
    add_noise_tf,
    random_intensity_tf,
    normalize_images_uniformly,
    split_data,
    adjust_image_shape,
    preprocessed_images,
    preprocessed_images_group
)


class TestBasicFunctions(unittest.TestCase):
    """
    Unit tests for basic image processing and augmentation functions
    defined in CNN_codes.utilities.

    Tests include padding, random rotation, zooming, noise addition,
    intensity scaling, and augmentation on 3D and 4D image data.
    """

    def setUp(self):
        """
        Setup test fixtures before each test method.

        Creates three synthetic 3D images of different shapes and stacks
        them into a padded 4D array for batch processing. Also defines
        dummy labels and target shape for testing augmentation functions.
        """
        logger.info("Setting up synthetic images and labels for basic tests.")
        self.img1 = np.ones((10, 12, 14))
        self.img2 = np.ones((8, 10, 12)) * 2
        self.img3 = np.ones((12, 8, 10)) * 3
        self.images = [self.img1, self.img2, self.img3]
        
        # Pad images to uniform shape for stacking and add channel dimension
        padded_images = pad_images(self.images)
        self.images_4d = np.expand_dims(padded_images, axis=-1)
        
        self.labels = np.array([0, 1, 1])
        self.target_shape = (16, 16, 16)
        logger.info("Setup complete.")

    def test_pad_images(self):
        """
        Test padding of images to the maximum shape among inputs.

        Asserts that the padded images have consistent shape with
        the maximum dimension size across all input images.
        """
        logger.info("Testing pad_images function.")
        padded = pad_images(self.images)
        self.assertEqual(padded.shape[1:], (12, 12, 14))
        logger.info("pad_images test passed.")

    def test_random_rotate_tf(self):
        """
        Test random 3D rotation on a TensorFlow tensor.

        Checks that output shape rank is preserved after rotation.
        """
        logger.info("Testing random_rotate_tf function.")
        tf_img = tf.convert_to_tensor(self.img1, dtype=tf.float32)
        rotated = random_rotate_tf(tf_img)
        self.assertEqual(rotated.shape.rank, 3)
        logger.info("random_rotate_tf test passed.")

    def test_random_zoom_and_crop_tf(self):
        """
        Test random zoom and crop augmentation on a 3D tensor.

        Verifies the output shape matches the target shape parameter.
        """
        logger.info("Testing random_zoom_and_crop_tf function.")
        tf_img = tf.convert_to_tensor(self.img1, dtype=tf.float32)
        zoomed = random_zoom_and_crop_tf(tf_img, self.target_shape)
        self.assertEqual(tuple(zoomed.shape), self.target_shape)
        logger.info("random_zoom_and_crop_tf test passed.")

    def test_add_noise_tf(self):
        """
        Test noise addition to a 3D tensor.

        Ensures output shape remains the same after adding noise.
        """
        logger.info("Testing add_noise_tf function.")
        tf_img = tf.convert_to_tensor(self.img1, dtype=tf.float32)
        noisy = add_noise_tf(tf_img, 0.1)
        self.assertEqual(noisy.shape, tf_img.shape)
        logger.info("add_noise_tf test passed.")

    def test_random_intensity_tf(self):
        """
        Test random intensity scaling of a 3D tensor.

        Confirms the shape is preserved after intensity scaling.
        """
        logger.info("Testing random_intensity_tf function.")
        tf_img = tf.convert_to_tensor(self.img1, dtype=tf.float32)
        scaled = random_intensity_tf(tf_img, 0.1)
        self.assertEqual(scaled.shape, tf_img.shape)
        logger.info("random_intensity_tf test passed.")


class TestDataProcessing(unittest.TestCase):
    """
    Unit tests for image normalization, splitting, and shape adjustment functions.
    """

    def test_normalize_images_uniformly(self):
        """
        Test normalization of image intensities to [0, 1].

        Checks that normalized images have min ~0 and max ~1 after scaling.
        """
        logger.info("Testing normalize_images_uniformly with random values.")
        array = np.random.rand(2, 4, 4, 4, 1) * 100
        result = normalize_images_uniformly(array)
        if isinstance(result, tuple):
            normed, *_ = result
        else:
            normed = result
        self.assertAlmostEqual(normed.min(), 0.0, delta=1e-5)
        self.assertAlmostEqual(normed.max(), 1.0, delta=1e-5)
        logger.info("normalize_images_uniformly test passed with random input.")

    def test_normalize_uniform_values(self):
        """
        Test normalization function on uniform image data.

        Confirms that an array of ones remains unchanged after normalization.
        """
        logger.info("Testing normalize_images_uniformly with uniform values.")
        array = np.ones((2, 4, 4, 4, 1))
        result = normalize_images_uniformly(array)
        if isinstance(result, tuple):
            normed = result[0]
        else:
            normed = result
        self.assertTrue(np.array_equal(normed, array))
        logger.info("normalize_images_uniformly test passed with uniform input.")

    def test_split_data(self):
        """
        Test data splitting into train, validation, and test sets.

        Checks that training data shape preserves original spatial dims and channels.
        """
        logger.info("Testing split_data function.")
        images = np.random.rand(100, 10, 10, 10, 1)
        labels = np.random.randint(0, 2, 100)
        x_train, y_train, x_val, y_val, x_test, y_test = split_data(images, labels)
        self.assertEqual(x_train.shape[1:], (10, 10, 10, 1))
        logger.info("split_data test passed.")

    def test_adjust_image_shape(self):
        """
        Test reshaping or cropping image batches to a target shape.

        Asserts the adjusted batch shape matches the requested dimensions.
        """
        logger.info("Testing adjust_image_shape function.")
        batch = np.random.rand(2, 20, 22, 18, 1)
        adjusted = adjust_image_shape(batch, (16, 16, 16))
        self.assertEqual(adjusted.shape, (2, 16, 16, 16, 1))
        logger.info("adjust_image_shape test passed.")


class TestPreprocessingWithMocks(unittest.TestCase):
    """
    Unit tests for preprocessed_images and preprocessed_images_group functions
    using mocking to simulate file IO and image loading.
    """

    @mock.patch('CNN_codes.utilities.nib.load')
    @mock.patch('CNN_codes.utilities.os.listdir')
    def test_preprocessed_images(self, mock_listdir, mock_nib_load):
        """
        Test preprocessed_images function with mocked filesystem and nibabel.

        Simulates loading two subjects with a fake atlas and verifies
        output image shape and dimensionality.
        """
        logger.info("Testing preprocessed_images with mocked file loading.")
        mock_listdir.return_value = ['s1.nii.gz', 's2.nii.gz']
        fake_atlas = np.zeros((10, 10, 10))
        fake_atlas[:, :, 5:7] = 165
        fake_image = np.ones((10, 10, 10))

        mock_img = mock.Mock()
        mock_img.get_fdata.return_value = fake_image
        mock_atlas = mock.Mock()
        mock_atlas.get_fdata.return_value = fake_atlas

        mock_nib_load.side_effect = lambda path: mock_atlas if 'atlas' in path else mock_img

        result = preprocessed_images('fake_folder', 'fake_atlas.nii.gz', roi_ids=(165,))
        self.assertEqual(result.shape[0], 2)
        self.assertEqual(result.ndim, 5)
  
if __name__ == '__main__':
    # Run all tests with verbosity, enabling loguru info logs to console
    logger.info("Starting all unittest executions.")
    unittest.main(verbosity=2)