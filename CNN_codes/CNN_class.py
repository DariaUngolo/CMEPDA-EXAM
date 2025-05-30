import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy

import tensorflow
from sklearn.metrics import roc_curve, auc
from keras.layers import MaxPooling3D, Conv3D, Flatten, Dense, BatchNormalization, Activation, Dropout
from keras.optimizers import SGD
from tensorflow.keras.regularizers import l2

from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.models import Model, Sequential
from keras.losses import BinaryCrossentropy
from keras.layers import Input

import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


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

        #Define the model architecture
        self.model = Sequential([
            Input(shape=input_shape),

            # MaxPooling before first Conv3D
            MaxPooling3D(pool_size=(2, 2, 2), padding='valid'),

            Conv3D(16, (3, 3, 3), activation='relu', padding='same'),
            #BatchNormalization(),



            Conv3D(32, (3, 3, 3), activation='relu', padding='same'),
            #BatchNormalization(),

            Flatten(),

            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        logger.info("Model architecture successfully initialized.")


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

        Parameters:
        ----------

        x_train, y_train : numpy.ndarray
            Training data and labels.
        x_val, y_val : numpy.ndarray
            Validation data and labels.
        x_test, y_test : numpy.ndarray
            Test data and labels.
        n_epochs : int
            Number of training epochs.
        batchsize : int
            Batch size for training.

        """

        logger.info("Starting training with %d epochs and batch size %d", n_epochs, batchsize)

        # Compile the model
        self.compile(
            optimizer=SGD(learning_rate=0.001),
            loss=BinaryCrossentropy(),
            metrics=['accuracy']
        )
        logger.debug("Model compiled successfully.")

        # Define callbacks
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

        early_stopping = EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=20,
            verbose=0,
            mode="auto",
            restore_best_weights=False,
            start_from_epoch=10
        )


        # Train the model
        history = self.fit(
            x_train, y_train,
            batch_size=batchsize,
            epochs=n_epochs,
            steps_per_epoch=round(len(x_train) / batchsize),
            verbose=1,
            validation_data=(x_val, y_val),
            validation_steps=round(len(x_val) / batchsize),
            callbacks=[reduce_on_plateau, early_stopping]
        )
        logger.info("Model training completed.")

        # Plot accuracy and loss
        self.accuracy_loss_plot(history)

        # Evaluate ROC
        self.validation_roc(x_val, y_val)
        self.test_roc(x_test, y_test)

        # Save model weights
        model_path = Path("model.h5")
        self.save_weights(model_path)
        logger.info("Model weights saved to %s", model_path)

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

        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, history.history['accuracy'], label="Train Accuracy")
        plt.plot(epochs_range, history.history['val_accuracy'], label="Validation Accuracy")
        plt.legend(loc="lower right")
        plt.title("Training and Validation Accuracy")

        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, history.history['loss'], label="Train Loss")
        plt.plot(epochs_range, history.history['val_loss'], label="Validation Loss")
        plt.legend(loc="upper right")
        plt.title("Training and Validation Loss")


        plt.show()
        logger.info("Accuracy and loss plots displayed.")

    def validation_roc(self, x_val, y_val):
        """
        Evaluates the model using ROC on validation data.

        Parameters:
        ----------
        x_val, y_val : numpy.ndarray
            Validation data and labels.

        """

        logger.info("Calculating ROC curve for validation data.")
        confidence_int = 0.683  # Livello di confidenza
        z_score = scipy.stats.norm.ppf((1 + confidence_int) / 2.0)  # Calcolo del punteggio z

        # Evaluate accuracy on validation data
        _, val_acc = self.evaluate(x_val, y_val, verbose=0)
        accuracy_err = z_score * np.sqrt((val_acc * (1 - val_acc)) / y_val.shape[0])
        logger.info(f"Validation Accuracy: {round(val_acc, 2)} ± {round(accuracy_err, 2)}")

        # Generate predictions and compute ROC curve
        preds = self.predict(x_val, verbose=1)
        fpr, tpr, _ = roc_curve(y_val, preds)
        roc_auc = auc(fpr, tpr)
        logger.info(f"Validation ROC AUC: {round(roc_auc, 2)}")

        # Calculate AUC confidence interval
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
        plt.show()


    def load(self, path, x_train, y_train, x_val, y_val, x_test, y_test, n_epochs, batchsize):
        """
        Loads pretrained weights and continues model training and evaluation.
    
        This method initializes the model with the required compilation parameters,
        loads pretrained weights from a specified path, and resumes the training
        process. It also evaluates the model performance on the provided datasets.
    
        Parameters:
        ----------
        path : str
            Path to the saved model weights.
        x_train : numpy.ndarray
            Training feature data.
        y_train : numpy.ndarray
            Training labels.
        x_val : numpy.ndarray
            Validation feature data.
        y_val : numpy.ndarray
            Validation labels.
        x_test : numpy.ndarray
            Test feature data.
        y_test : numpy.ndarray
            Test labels.
        n_epochs : int
            Number of epochs for continued training.
        batchsize : int
            Batch size for training.
    
        """
        # Compile the model with initial parameters
        logger.info("Compiling the model with initial parameters.")
        self.compile(optimizer=SGD(learning_rate=0.01), loss=BCE(), metrics=['accuracy'])
    
        # Perform a single training step
        logger.info("Performing a single training step to initialize model weights.")
        self.train_on_batch(x_train, y_train)
    
        # Load pretrained weights from the specified path
        logger.info(f"Loading pretrained weights from: {path}")
        self.load_weights(path)
    
        # Continue training and evaluate the model on the provided datasets
        logger.info("Resuming training and evaluating the model.")
        self.compile_and_fit(x_train, y_train, x_val, y_val, x_test, y_test, n_epochs, batchsize)
