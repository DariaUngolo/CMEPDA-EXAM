import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy

import tensorflow
from sklearn.metrics import roc_curve, auc
from sklearn.utils import class_weight
from keras.layers import (MaxPooling3D, Conv3D, Flatten, Dense,Input, BatchNormalization,LeakyReLU, Activation, Dropout,PReLU, ReLU, GlobalAveragePooling3D)
from keras.optimizers import SGD

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.models import load_model, Model, Sequential
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.initializers import HeNormal
import random

from keras.losses import BinaryCrossentropy, MeanSquaredError
import seaborn as sns


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

        logger.info(f"Initializing MyCNNModel with input shape: {input_shape}")
        super(MyCNNModel, self).__init__()

        # Define the model architecture using Keras Sequential API
        self.model = Sequential([

            # First block
            Conv3D(8, kernel_size=3,kernel_initializer=HeNormal(), activation=None, padding='same',kernel_regularizer=l1(0.001)),
            BatchNormalization(),
            ReLU(),
            MaxPooling3D(pool_size=(2, 2, 2)),
            Dropout(0.3),

            # Second block
            Conv3D(16, kernel_size=3, activation=None, padding='same', kernel_regularizer=l1(0.001)),
            BatchNormalization(),
            ReLU(),
            MaxPooling3D(pool_size=(2, 2, 2)),
            Dropout(0.3),

            # Third block
            Conv3D(32, kernel_size=3, activation=None, padding='same', kernel_regularizer=l1(0.001)),
            ReLU(),
            MaxPooling3D(pool_size=(2, 2, 2)),
            Dropout(0.3),

            # Fourth block (extra layer)
            Conv3D(32, kernel_size=3, activation=None, padding='same', kernel_regularizer=l1(0.001)),
            ReLU(),
            Dropout(0.4),

            #Flatten
            Flatten(),

            # Fully connected layers
            Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            Dense(1, activation='sigmoid')  # Binary classification
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
            optimizer=Adam(learning_rate=0.001),
            loss=BinaryCrossentropy(),
            metrics=['accuracy', tensorflow.keras.metrics.AUC(), tensorflow.keras.metrics.Recall()]
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
                patience=30,
                verbose=1,
                mode="auto",
                restore_best_weights=True,
                start_from_epoch=30
        )



        # Prepare TensorFlow datasets with batching and prefetching for performance
        train_dataset = tensorflow.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train)).batch(batchsize).repeat().prefetch(tensorflow.data.AUTOTUNE)
        val_dataset = tensorflow.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batchsize).prefetch(tensorflow.data.AUTOTUNE)
        test_dataset = tensorflow.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batchsize).prefetch(tensorflow.data.AUTOTUNE)


        weights = class_weight.compute_class_weight(class_weight='balanced',
                                                    classes=np.unique(y_train),
                                                    y=y_train)

        class_weights = {0: weights[0], 1: weights[1]}


        # Train the model
        history = self.fit(
                train_dataset,
                #batch_size=batchsize,
                epochs=n_epochs,
                steps_per_epoch=(len(x_train) // batchsize),
                verbose=2,
                validation_data=val_dataset,
                validation_steps=round(len(x_val) // batchsize),
                callbacks=[reduce_on_plateau,early_stopping],
                class_weight=class_weights
        )
        logger.info("Model training completed.")

        # Plot accuracy and loss
        self.AUC_loss_plot(history)


        # Extract all validation and test data batches for ROC calculation
        x_val, y_val = self.extract_data_and_labels(val_dataset)
        x_test, y_test = self.extract_data_and_labels(test_dataset)


        # Evaluate model performance on validation and test datasets using ROC curve
        self.compute_and_plot_roc(x_val, y_val, dataset_name="validation")
        self.compute_and_plot_roc(x_test, y_test, dataset_name="test")




    def AUC_loss_plot(self, history):
        """
        Plots training and validation AUC and loss curves.

        Parameters:

        ----------
        history : keras.callbacks.History
            Training history object returned by `fit`.

        """

        logger.info("Plotting training and validation AUC and loss.")
        epochs_range = range(1, len(history.history['auc']) + 1)

        # Update matplotlib configuration for high-quality, compact plots
        plt.rcParams.update({
            'font.size': 6,
            'font.family': 'serif',
            'axes.labelsize': 5,
            'axes.titlesize': 6,
            'legend.fontsize': 4,
            'xtick.labelsize': 4,
            'ytick.labelsize': 4,
            'axes.grid': True,
            'grid.alpha': 0.2,
            'grid.linestyle': '--',
            'lines.linewidth': 0.8,
            'figure.dpi': 600,
            'savefig.dpi': 600
        })

        # Use colorblind-safe palette
        sns.set_palette("colorblind")

        logger.info("Generating training history plots...")

        # Create the figure
        plt.figure(figsize=(4, 2))  # compact high-quality layout

        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, history.history['auc'], label="Train AUC")
        plt.plot(epochs_range, history.history['val_auc'], label="Validation AUC")
        plt.xlabel("Epochs", labelpad=2, fontweight='semibold')
        plt.ylabel("AUC", labelpad=2, fontweight='semibold')
        plt.title("Training and Validation AUC", fontweight='bold', pad=4)
        plt.legend(loc="best", frameon=False )
        plt.grid(axis='both', linestyle='--', linewidth=0.3, alpha=0.2)
        plt.tick_params(axis='both', direction='in', length=2, width=0.3)

        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, history.history['loss'], label="Train Loss")
        plt.plot(epochs_range, history.history['val_loss'], label="Validation Loss")
        plt.xlabel("Epochs", labelpad=2, fontweight='semibold')
        plt.ylabel("Loss", labelpad=2, fontweight='semibold')
        plt.title("Training and Validation Loss", fontweight='bold', pad=4)
        plt.legend(loc="best", frameon=False)
        plt.grid(axis='both', linestyle='--', linewidth=0.3, alpha=0.2)
        plt.tick_params(axis='both', direction='in', length=2, width=0.3)

        # Improve layout and aesthetics
        sns.despine(trim=True)
        plt.tight_layout()

        # Save plot to current script directory
        save_name = "AUC_loss_train+val.png"
        base_path = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(base_path, save_name)

        plt.savefig(save_path)
        plt.show()

        logger.info("AUC and loss plots displayed.")

    def compute_and_plot_roc(self, x_data, y_data, dataset_name="validation"):
        """
            Evaluates the model using ROC on provided dataset.

            Parameters
            ----------
            x_data : np.ndarray
                Feature data (e.g., validation or test).
            y_data : np.ndarray
                True binary labels.
            dataset_name : str
                Name of the dataset for labeling/logging (e.g., "validation", "test").
            """

        logger.info(f"Calculating ROC curve for {dataset_name} data.")
        confidence_int = 0.683  # Confidence interval for accuracy error bars
        z_score = scipy.stats.norm.ppf((1 + confidence_int) / 2.0)

        # Evaluate accuracy on validation data
        results = self.evaluate(x_data, y_data, verbose=0)
        _, acc, auc_value, recall = results
        accuracy_err = z_score * np.sqrt((acc * (1 - acc)) / y_data.shape[0])
        recall_err = z_score * np.sqrt((recall * (1 - recall)) / y_data.shape[0])
        logger.info(f"{dataset_name} Accuracy: {round(acc, 2)} ± {round(accuracy_err, 2)}")
        logger.info(f"{dataset_name} Recall: {round(recall, 2)} ± {round(recall_err, 2)}")



        # Generate predictions and compute ROC curve
        preds = self.predict(x_data, verbose=1)
        fpr, tpr, _ = roc_curve(y_data, preds)
        roc_auc = auc(fpr, tpr)
        logger.info(f"{dataset_name} ROC AUC: {round(roc_auc, 2)}")

        # Calculate standard error for AUC
        n1 = np.sum(y_data == 1)
        n2 = np.sum(y_data == 0)
        q1 = roc_auc / (2 - roc_auc)
        q2 = 2 * roc_auc ** 2 / (1 + roc_auc)
        auc_err = z_score * np.sqrt(
            (roc_auc * (1 - roc_auc) + (n1 - 1) * (q1 - roc_auc ** 2) + (n2 - 1) * (q2 - roc_auc ** 2)) / (n1 * n2)
        )
        logger.info(f"{dataset_name} AUC: {round(roc_auc, 2)} ± {round(auc_err, 2)}")

        # Update matplotlib configuration for high-quality, compact plots
        plt.rcParams.update({
            'font.size': 6,
            'font.family': 'serif',
            'axes.labelsize': 5,
            'axes.titlesize': 6,
            'legend.fontsize': 4,
            'xtick.labelsize': 4,
            'ytick.labelsize': 4,
            'axes.grid': True,
            'grid.alpha': 0.2,
            'grid.linestyle': '--',
            'lines.linewidth': 0.8,
            'figure.dpi': 600,
            'savefig.dpi': 600
        })

        # Use colorblind-safe palette
        sns.set_palette("colorblind")

        # Plot ROC curve
        plt.figure(figsize=(2, 1.4))
        plt.plot(fpr, tpr, color='tab:blue', lw=0.8, label=f'ROC curve (area = {roc_auc:.2f}  ± {auc_err:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=0.6)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', labelpad=2, fontweight='semibold')
        plt.ylabel('True Positive Rate', labelpad=2, fontweight='semibold')
        plt.title(f'{dataset_name} ROC', fontweight='bold', pad=4)
        plt.legend(loc="lower right", frameon=False)
        plt.grid(axis='both', linestyle='--', linewidth=0.3, alpha=0.2)
        plt.tick_params(axis='both', direction='in', length=2, width=0.3)
        sns.despine(trim=True)
        plt.tight_layout()

        # Save image to current script directory
        base_path = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(base_path, f'CNN_{dataset_name}_roc.png')
        plt.savefig(save_path)

        plt.show()
        print(f"[DEBUG] {dataset_name}_roc: plot saved")




    def save_model(self, path):
        """
        Saves the entire model, including architecture, weights, and optimizer state, to a file.

        Parameters:
        -----------
        path : str, optional
            The file path where the model will be saved. Default is "model_full.h5".

        """
        self.model.save(path)
        logger.info(f"Model saved to {path}")


    def load_model(self, path):
        """
        Loads a complete model (architecture + weights + optimizer state) from a file.

        Parameters:
        -----------
        path : str, optional
            The file path from which to load the model. Default is "model_full.h5".

        """
        self.model = load_model(path)
        logger.info(f"Model loaded from {path}")




