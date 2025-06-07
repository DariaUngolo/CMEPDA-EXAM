import unittest
import numpy as np
import os
import tempfile
import tensorflow as tf
from loguru import logger

# Allow import from parent directory
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from CNN_codes.CNN_class import MyCNNModel

class TestMyCNNModel(unittest.TestCase):
    """
    Unit tests for the MyCNNModel class defined in CNN_class.py.

    Tests cover:
    - Model construction verification
    - Forward pass output shape correctness
    - Model saving and loading functionality
    - Extraction of data and labels from tf.data.Dataset
    - Training pipeline on synthetic data for crash-free execution
    """

    def setUp(self):
        """
        Setup method called before each test.

        Initializes a MyCNNModel instance with a reduced input shape for
        faster testing. Also generates synthetic input data and binary
        labels, split into training, validation, and test subsets.
        """
        logger.info("Setting up the test case with synthetic data and model initialization.")
        self.input_shape = (16, 16, 16, 1)  # Smaller shape for faster tests
        self.model = MyCNNModel(input_shape=self.input_shape)

        # Create a small synthetic dataset with random floats and binary labels
        self.num_samples = 20
        self.x_data = np.random.rand(self.num_samples, *self.input_shape).astype(np.float32)
        self.y_data = np.random.randint(0, 2, size=(self.num_samples, 1)).astype(np.float32)

        # Split dataset into training, validation, and testing partitions
        self.x_train, self.y_train = self.x_data[:10], self.y_data[:10]
        self.x_val, self.y_val     = self.x_data[10:15], self.y_data[10:15]
        self.x_test, self.y_test   = self.x_data[15:], self.y_data[15:]
        logger.info("Setup complete: data and model ready.")

    def test_model_construction(self):
        """
        Verify that the internal Keras model is correctly instantiated.

        Checks if the 'model' attribute of MyCNNModel is an instance of
        tf.keras.Sequential, indicating proper model construction.
        """
        logger.info("Testing model construction.")
        self.assertIsInstance(self.model.model, tf.keras.Sequential)
        logger.info("Model construction test passed.")

    def test_model_forward_pass(self):
        """
        Test the forward pass of the model on synthetic input data.

        Converts training data to a TensorFlow tensor, feeds it through
        the model, and checks that output shape matches expected batch size
        and has a single output unit (binary classification).
        """
        logger.info("Testing model forward pass.")
        x = tf.convert_to_tensor(self.x_train)
        y_pred = self.model(x)
        self.assertEqual(y_pred.shape[0], self.x_train.shape[0])
        self.assertEqual(y_pred.shape[-1], 1)
        logger.info("Model forward pass test passed with output shape %s.", y_pred.shape)

    def test_model_save_and_load(self):
        """
        Test saving the model to disk and loading it back.

        Uses a temporary directory to save the model in HDF5 format,
        then reloads it into a new MyCNNModel instance. Verifies the file
        exists after saving and that the loaded model is a tf.keras.Model.
        """
        logger.info("Testing model save and load functionality.")
        with tempfile.TemporaryDirectory() as tmpdirname:
            save_path = os.path.join(tmpdirname, 'temp_model.h5')
            # Run a forward pass with zeros to initialize weights before saving
            dummy = np.zeros((1, *self.input_shape), dtype=np.float32)
            _ = self.model(dummy)
            self.model.save_model(path=save_path)
            self.assertTrue(os.path.exists(save_path))
            logger.info("Model saved successfully at %s.", save_path)

            new_model = MyCNNModel(self.input_shape)
            new_model.load_model(path=save_path)
            self.assertIsInstance(new_model.model, tf.keras.Model)
            logger.info("Model loaded successfully from %s.", save_path)

    def test_extract_data_and_labels(self):
        """
        Test extracting numpy arrays of inputs and labels from a tf.data.Dataset.

        Converts training data into a batched tf.data.Dataset, then extracts
        all data and labels as tensors via the model's method. Checks that the
        shapes of the extracted arrays match the original training data.
        """
        logger.info("Testing data and labels extraction from tf.data.Dataset.")
        dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train)).batch(2)
        x_out, y_out = self.model.extract_data_and_labels(dataset)
        self.assertEqual(x_out.shape[0], self.x_train.shape[0])
        self.assertEqual(y_out.shape[0], self.y_train.shape[0])
        logger.info("Data and labels extraction test passed with shapes: %s, %s.", x_out.shape, y_out.shape)

    def test_compile_and_fit_on_synthetic_data(self):
        """
        Test the full training pipeline on small synthetic data.

        Calls the model's compile_and_fit method with few epochs and a small
        batch size to verify training completes without errors. After training,
        checks that predictions on test data have the correct shape.
        """
        logger.info("Testing model training pipeline on synthetic data.")
        self.model.compile_and_fit(
            x_train=self.x_train,
            y_train=self.y_train,
            x_val=self.x_val,
            y_val=self.y_val,
            x_test=self.x_test,
            y_test=self.y_test,
            n_epochs=2,            # Few epochs for speed
            batchsize=2            # Small batch size
        )
        preds = self.model(self.x_test)
        self.assertEqual(preds.shape[0], self.x_test.shape[0])
        logger.info("Training pipeline test passed with predictions shape: %s.", preds.shape)

if __name__ == '__main__':
    logger.info("Starting unit tests for MyCNNModel...")
    unittest.main()
