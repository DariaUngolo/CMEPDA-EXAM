import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy

import tensorflow
from sklearn.metrics import roc_curve, auc
from keras.layers import (MaxPooling3D, Conv3D, Flatten, Dense,Input, BatchNormalization, Activation, Dropout, GlobalAveragePooling3D)
from keras.optimizers import SGD

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model, Model, Sequential
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.losses import BinaryCrossentropy, MeanSquaredError


from loguru import logger

# Configure logger: you can customize the format, level, output file, etc.
logger.remove()  # Remove default logger to customize
logger.add(lambda msg: print(msg, end=''), colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")




strategy = tensorflow.distribute.MirroredStrategy()
logger.info(f"Number of GPUs detected: {strategy.num_replicas_in_sync}")


class MyCNNModel(tensorflow.keras.Model):
    """

    A lightweight 3D CNN model for binary classification.


    Attributes:
    -----------

    model : tensorflow.keras.Sequential
        The internal sequential model that defines the architecture.

    Methods:
    --------

    compile_and_fit:
        Compiles and trains the model on the provided datasets.
    accuracy_loss_plot:
        Plots training and validation accuracy and loss curves.
    validation_roc:
        Evaluates model performance using ROC on validation data.
    test_roc:
        Evaluates model performance using ROC on test data.
    load:
        Loads saved model weights and optionally continues training.

    """

    def __init__(self, input_shape):
        """

        Initializes the CNN model with a predefined architecture.

        Parameters:

        ----------

        input_shape : tuple, optional
            The shape of the input data

        """

        logger.info("Initializing MyCNNModel with input shape: %s", input_shape)
        super(MyCNNModel, self).__init__()

        # Define the model architecture using Keras Sequential API
        self.model =  Sequential([
            Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=tensorflow.keras.regularizers.l2(1e-4)),
            BatchNormalization(),
            MaxPooling3D(pool_size=(2, 2, 2)),
            Dropout(0.2),


            Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=tensorflow.keras.regularizers.l2(1e-4)),
            BatchNormalization(),
            MaxPooling3D(pool_size=(2, 2, 2)),
            Dropout(0.3),


            Flatten(),
            Dense(128, activation='relu', kernel_regularizer=tensorflow.keras.regularizers.l2(1e-4)),
            Dropout(0.4),
            Dense(1, activation='sigmoid')
        ])

        logger.info("Model architecture successfully initialized.")

    def extract_data_and_labels(self, dataset):
        """
        Extracts and concatenates all features and labels from a TensorFlow dataset.

        Parameters
        ----------
        dataset : tf.data.Dataset
            Dataset yielding tuples of (features, labels).

        Returns
        -------
        tf.Tensor
            Concatenated features tensor.
        tf.Tensor
            Concatenated labels tensor.

        """
        x_list = []
        y_list = []
        for x_batch, y_batch in dataset:
            x_list.append(x_batch)
            y_list.append(y_batch)
        x = tensorflow.concat(x_list, axis=0)
        y = tensorflow.concat(y_list, axis=0)

        return x, y

    def call(self, inputs, training=False):
        """

        Forward pass for the model.

        Parameters:

        ----------
        inputs : tensor
            Input tensor for the forward pass.
        training : bool, optional
            Whether the model is in training mode, by default False.

        Returns:

        -------
        tensor
            Output of the model.

        """
        return self.model(inputs, training=training)

    def compile_and_fit(self, x_train, y_train, x_val, y_val, x_test, y_test, n_epochs, batchsize):
        """

        Compiles the model and trains it on the provided datasets.

        Parameters
        ----------
        x_train : np.ndarray
            Training feature data.
        y_train : np.ndarray
            Training labels.
        x_val : np.ndarray
            Validation feature data.
        y_val : np.ndarray
            Validation labels.
        x_test : np.ndarray
            Test feature data.
        y_test : np.ndarray
            Test labels.
        n_epochs : int
            Number of training epochs.
        batchsize : int
            Batch size for training.

        """

        logger.info(f"Starting training with {n_epochs} epochs and batch size {batchsize}")

        self.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss=BinaryCrossentropy(),
            metrics=['accuracy']
        )

        logger.debug("Model compiled successfully.")

        # Learning rate scheduler callback
        reduce_on_plateau = ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.1,
                patience=20,
                verbose=0,
                mode="auto",
                min_delta=0.0001,
                cooldown=0,
                min_lr=0
        )

        # Early stopping callback
        early_stopping = EarlyStopping(
                monitor="val_loss",
                min_delta=0,
                patience=20,
                verbose=1,
                mode="auto",
                restore_best_weights=False,
                start_from_epoch=10
        )

        # Prepare TensorFlow datasets with batching and prefetching for performance
        train_dataset = tensorflow.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train)).batch(batchsize).repeat().prefetch(tensorflow.data.AUTOTUNE)
        val_dataset = tensorflow.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batchsize).prefetch(tensorflow.data.AUTOTUNE)
        test_dataset = tensorflow.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batchsize).prefetch(tensorflow.data.AUTOTUNE)

        # Extract all validation and test data batches for ROC calculation
        x_val, y_val = self.extract_data_and_labels(val_dataset)
        x_test, y_test = self.extract_data_and_labels(test_dataset)

        # Train the model
        history = self.fit(
                train_dataset,
                batch_size=batchsize,
                epochs=n_epochs,
                steps_per_epoch=round(len(x_train) // batchsize),
                verbose=1,
                validation_data=val_dataset,
                validation_steps=round(len(x_val) // batchsize),
                callbacks=[reduce_on_plateau]
        )
        logger.info("Model training completed.")

        # Plot accuracy and loss
        self.accuracy_loss_plot(history)


        # Extract all validation and test data batches for ROC calculation
        x_val, y_val = self.extract_data_and_labels(val_dataset)
        x_test, y_test = self.extract_data_and_labels(test_dataset)


        # Evaluate model performance on validation and test datasets using ROC curve
        self.validation_roc(x_val, y_val)
        self.test_roc(x_test, y_test)

        # Salva il modello completo subito dopo il training
        #self.save_model("model_full.h5")



    def accuracy_loss_plot(self, history):
        """
        Plots training and validation accuracy and loss curves.

        Parameters:

        ----------
        history : keras.callbacks.History
            Training history object returned by `fit`.

        """

        logger.info("Plotting training and validation accuracy and loss.")
        epochs_range = range(1, len(history.history['accuracy']) + 1)
        plt.figure(figsize=(15, 15))

        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, history.history['accuracy'], label="Train Accuracy")
        plt.plot(epochs_range, history.history['val_accuracy'], label="Validation Accuracy")
        plt.legend(loc="lower right")
        plt.title("Training and Validation Accuracy")

        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, history.history['loss'], label="Train Loss")
        plt.plot(epochs_range, history.history['val_loss'], label="Validation Loss")
        plt.legend(loc="upper right")
        plt.title("Training and Validation Loss")

        # Save plot to current script directory
        save_name="training_plot.png"
        base_path = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(base_path, save_name)

        plt.savefig(save_path)
        plt.show()
        logger.info("Accuracy and loss plots displayed.")


        plt.show()
        logger.info("Accuracy and loss plots displayed.")

    def validation_roc(self, x_val, y_val):
        """
        Evaluates the model using ROC on validation data.

        Parameters
        ----------
        x_val : np.ndarray
            Validation feature data.
        y_val : np.ndarray
            Validation labels.

        """

        logger.info("Calculating ROC curve for validation data.")
        confidence_int = 0.683  # Confidence interval for accuracy error bars
        z_score = scipy.stats.norm.ppf((1 + confidence_int) / 2.0)

        # Evaluate accuracy on validation data
        _, val_acc = self.evaluate(x_val, y_val, verbose=0)
        accuracy_err = z_score * np.sqrt((val_acc * (1 - val_acc)) / y_val.shape[0])
        logger.info(f"Validation Accuracy: {round(val_acc, 2)} ± {round(accuracy_err, 2)}")

        # Generate predictions and compute ROC curve
        preds = self.predict(x_val, verbose=1)
        fpr, tpr, _ = roc_curve(y_val, preds)
        roc_auc = auc(fpr, tpr)
        logger.info(f"Validation ROC AUC: {round(roc_auc, 2)}")

        # Calculate standard error for AUC
        n1 = np.sum(y_val == 1)
        n2 = np.sum(y_val == 0)
        q1 = roc_auc / (2 - roc_auc)
        q2 = 2 * roc_auc ** 2 / (1 + roc_auc)
        auc_err = z_score * np.sqrt(
            (roc_auc * (1 - roc_auc) + (n1 - 1) * (q1 - roc_auc ** 2) + (n2 - 1) * (q2 - roc_auc ** 2)) / (n1 * n2)
        )
        logger.info(f"Validation AUC: {round(roc_auc, 2)} ± {round(auc_err, 2)}")

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Validation ROC')
        plt.legend(loc="lower right")

        # Salvataggio immagine nella stessa cartella dello script
        base_path = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(base_path, 'validation_roc.png')
        plt.savefig(save_path)

        plt.show()
        print("[DEBUG] validation_roc: plot mostrato")

    def test_roc(self, x_test, y_test):

        """

        Evaluates the model's performance on test data using ROC analysis.

        This method computes the ROC curve and its corresponding area under the curve (AUC)
        for the given test dataset. It also calculates the confidence intervals
        for accuracy and AUC based on the given confidence level.

        Parameters:

        ----------

        x_test : numpy.ndarray
            Test feature data.
        y_test : numpy.ndarray
            Test labels.

        """
        logger.info("Starting ROC curve computation for test data.")
        confidence_int = 0.683  # Confidence level
        z_score = scipy.stats.norm.ppf((1 + confidence_int) / 2.0)

        # Evaluate accuracy on test data
        test_loss, test_acc = self.evaluate(x_test, y_test, verbose=0)
        accuracy_err = z_score * np.sqrt((test_acc * (1 - test_acc)) / y_test.shape[0])
        logger.info(f"Test Accuracy: {round(test_acc, 2)} ± {round(accuracy_err, 2)}")

        # Generate predictions and compute ROC curve
        preds_test = self.predict(x_test, verbose=1)
        fpr, tpr, _ = roc_curve(y_test, preds_test)
        roc_auc = auc(fpr, tpr)
        logger.info(f"Test ROC AUC: {round(roc_auc, 2)}")

        # Calculate AUC confidence interval
        n1 = np.sum(y_test == 1)
        n2 = np.sum(y_test == 0)
        q1 = roc_auc / (2 - roc_auc)
        q2 = 2 * roc_auc ** 2 / (1 + roc_auc)
        auc_err = z_score * np.sqrt(
            (roc_auc * (1 - roc_auc) + (n1 - 1) * (q1 - roc_auc ** 2) + (n2 - 1) * (q2 - roc_auc ** 2)) / (n1 * n2)
        )
        logger.info(f"Test AUC: {round(roc_auc, 2)} ± {round(auc_err, 2)}")

        # Plot ROC curve
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Test ROC')
        plt.legend(loc="lower right")

        # salva immagine
        base_path = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(base_path, 'test_roc.png')
        plt.savefig(save_path)
        print(f"[DEBUG] test_roc: plot salvato in {save_path}")

        plt.show()

    def save_model(self, path="model_full.h5"):
        """
        Saves the entire model, including architecture, weights, and optimizer state, to a file.

        Parameters:
        -----------
        path : str, optional
            The file path where the model will be saved. Default is "model_full.h5".

        """
        self.model.save(path)
        logger.info(f"Model saved to {path}")


    def load_model(self, path="model_full.h5"):
        """
        Loads a complete model (architecture + weights + optimizer state) from a file.

        Parameters:
        -----------
        path : str, optional
            The file path from which to load the model. Default is "model_full.h5".

        """
        self.model = load_model(path)
        logger.info(f"Model loaded from {path}")




