import sys
import os
from pathlib import Path
import scipy
import matplotlib.pyplot as plt
import tensorflow
from sklearn.metrics import roc_curve, auc
import numpy as np
from keras.layers import MaxPooling3D, Conv3D, Flatten, Dense, BatchNormalization, Activation, Dropout
from keras.optimizers import SGD
from tensorflow.keras.regularizers import l2

from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.models import Model, Sequential
from keras.losses import BinaryCrossentropy
from keras.layers import Input


class MyCNNModel(tensorflow.keras.Model):

    def __init__(self, input_shape=(121, 145, 47, 1)):

        print("[DEBUG] __init__: inizializzazione modello leggero con input_shape:", input_shape)
        super(MyCNNModel, self).__init__()

        self.model = Sequential()

         # Primo livello: MaxPooling3D
        self.model.add(Input(shape=input_shape))
        print("[DEBUG] __init__: Input layer aggiunto")
        self.model.add(MaxPooling3D(pool_size=(2, 2, 2), padding="valid"))
        print("[DEBUG] __init__: Primo MaxPooling3D aggiunto")

        # Secondo livello: Conv3D con kernel_regularizer
        self.model.add(Conv3D(filters=16, kernel_size=(3, 3, 3), padding='valid', activation='relu'))
        print("[DEBUG] __init__: Primo Conv3D aggiunto")
        self.model.add(BatchNormalization())

        # Terzo livello: MaxPooling3D
        self.model.add(MaxPooling3D(pool_size=(2, 2, 2)))
        print("[DEBUG] __init__: Secondo MaxPooling3D aggiunto")

        # Quarto livello: Conv3D con regolarizzazione
        self.model.add(Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu'))
        print("[DEBUG] __init__: Secondo Conv3D aggiunto")
        self.model.add(BatchNormalization())
        

        # Flatten e Dense layers
        self.model.add(Flatten())
        print("[DEBUG] __init__: Flatten aggiunto")
        self.model.add(Dropout(0.5))
        self.model.add(Dense(units=64, activation='relu', kernel_regularizer=l2(0.01)))
        print("[DEBUG] __init__: Dense 64 aggiunto")

        # Layer finale
        self.model.add(Dense(units=1, activation='sigmoid'))
        print("[DEBUG] __init__: Dense finale aggiunto")


    def call(self, inputs, training=False):
        # Chiama il modello sequenziale interno
        return self.model(inputs, training=training)

    def compile_and_fit(self, x_train, y_train, x_val, y_val, x_test, y_test, n_epochs, batchsize):
        print("[DEBUG] compile_and_fit: inizio")
        print(f"[DEBUG] compile_and_fit: x_train shape {x_train.shape}, y_train shape {y_train.shape}")
        print(f"[DEBUG] compile_and_fit: x_val shape {x_val.shape}, y_val shape {y_val.shape}")
        print(f"[DEBUG] compile_and_fit: x_test shape {x_test.shape}, y_test shape {y_test.shape}")
        print(f"[DEBUG] compile_and_fit: n_epochs {n_epochs}, batchsize {batchsize}")

        # Compilazione del modello
        self.compile(
            optimizer=SGD(learning_rate=0.01),  # Ottimizzatore SGD con learning rate 0.01
            loss=BinaryCrossentropy(),                      # Funzione di perdita BCE
            metrics=['accuracy']               # Metri: accuratezza
        )
        print("[DEBUG] compile_and_fit: modello compilato")

        # Callback per migliorare l'allenamento
        reduce_on_plateau = ReduceLROnPlateau(
            monitor="val_loss",   # Controlla la perdita sui dati di validazione
            factor=0.1,           # Riduce il learning rate di 10 volte se non migliora
            patience=20,          # Aspetta 20 epoche senza miglioramento
            verbose=0,
            mode="auto",
            min_delta=0.0001,     # Miglioramento minimo richiesto
            cooldown=0,
            min_lr=0              # Limite inferiore del learning rate
        )

        early_stopping = EarlyStopping(
            monitor="val_loss",          # Controlla la perdita sui dati di validazione
            min_delta=0,                 # Nessun miglioramento richiesto oltre il valore corrente
            patience=20,                 # Termina l'allenamento se non ci sono miglioramenti per 20 epoche
            verbose=0,
            mode="auto",
            restore_best_weights=False,  # Non ripristina i migliori pesi
            start_from_epoch=10          # Considera l'interruzione solo dopo 10 epoche
        )

        # Parametri di addestramento
        epochs = n_epochs        # Numero di epoche totali (passaggi completi sul dataset)
        batch_size = batchsize   # Dimensione del batch (numero di campioni per passo)

        # Avvio del processo di allenamento
        history = self.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=round(len(x_train) / batch_size),  # Numero di passi per epoca
            verbose=1,                                         # Mostra dettagli durante l'allenamento
            validation_data=(x_val, y_val),                   # Dati di validazione
            validation_steps=round(len(x_val) / batch_size),  # Numero di passi per validazione
            callbacks=[reduce_on_plateau, early_stopping]     # Callback definiti sopra
        )
        print("[DEBUG] compile_and_fit: fit completato")

        # Visualizzazione dei risultati
        self.accuracy_loss_plot(history)  # Disegna i grafici di accuratezza e perdita
        print("[DEBUG] compile_and_fit: plot di accuracy e loss mostrati")

        # Valutazione ROC su validazione e test
        self.validation_roc(x_val, y_val)
        print("[DEBUG] compile_and_fit: validazione ROC completata")
        self.test_roc(x_test, y_test)
        print("[DEBUG] compile_and_fit: test ROC completata")

        # Salvataggio dei pesi del modello
        self.save_weights(Path('C:\\Users\\daria\\OneDrive\\Desktop\\model.h5'))  # Pesa salvati in "model.h5"
        print("[DEBUG] compile_and_fit: pesi salvati")

    def accuracy_loss_plot(self, history):
        print("[DEBUG] accuracy_loss_plot: inizio")
        # Estrazione dei dati di accuratezza e perdita dall'oggetto history
        acc = history.history['accuracy']         # Accuratezza su dati di addestramento
        val_acc = history.history['val_accuracy'] # Accuratezza su dati di validazione
        loss = history.history['loss']            # Perdita su dati di addestramento
        val_loss = history.history['val_loss']    # Perdita su dati di validazione
        print(f"[DEBUG] accuracy_loss_plot: {len(acc)} epoche trovate")

        # Creazione di una sequenza di epoche per l'asse X
        epochs_range = range(1, len(acc) + 1)

        # Imposta la dimensione complessiva della figura
        plt.figure(figsize=(15, 15))

        # Grafico 1: Accuratezza di addestramento e validazione
        plt.subplot(1, 2, 1)  # Una griglia di 1x2 con questo come primo grafico
        plt.plot(epochs_range, acc, label='Training Accuracy')  # Accuratezza addestramento
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')  # Accuratezza validazione
        plt.legend(loc='lower right')  # Posizione della legenda
        plt.title('Training and Validation Accuracy')  # Titolo del grafico

        # Grafico 2: Perdita di addestramento e validazione
        plt.subplot(1, 2, 2)  # Secondo grafico nella griglia
        plt.plot(epochs_range, loss, label='Training Loss')  # Perdita addestramento
        plt.plot(epochs_range, val_loss, label='Validation Loss')  # Perdita validazione
        plt.legend(loc='upper right')  # Posizione della legenda
        plt.title('Training and Validation Loss')  # Titolo del grafico

        # Mostra i grafici creati
        plt.show()
        print("[DEBUG] accuracy_loss_plot: grafici mostrati")

    def validation_roc(self, x_val, y_val):
        print("[DEBUG] validation_roc: inizio")
        confidence_int = 0.683  # Livello di confidenza
        z_score = scipy.stats.norm.ppf((1 + confidence_int) / 2.0)  # Calcolo del punteggio z

        # Valutazione dell'accuratezza
        _, val_acc = self.evaluate(x_val, y_val, verbose=0)
        print(f"[DEBUG] validation_roc: val_acc={val_acc}")
        accuracy_err = z_score * np.sqrt((val_acc * (1 - val_acc)) / y_val.shape[0])
        print(f'Validation accuracy: {round(val_acc, 2)} +/- {round(accuracy_err, 2)}')

        # Predizioni e calcolo della curva ROC
        preds = self.predict(x_val, verbose=1)
        print(f"[DEBUG] validation_roc: preds shape {preds.shape}, min={preds.min()}, max={preds.max()}")
        fpr, tpr, _ = roc_curve(y_val, preds)
        roc_auc = auc(fpr, tpr)
        print(f"[DEBUG] validation_roc: roc_auc={roc_auc}")

        # Calcolo dell'errore sull'AUC
        n1 = np.sum(y_val == 1)
        n2 = np.sum(y_val == 0)
        q1 = roc_auc / (2 - roc_auc)
        q2 = 2 * roc_auc ** 2 / (1 + roc_auc)
        auc_err = z_score * np.sqrt(
            (roc_auc * (1 - roc_auc) + (n1 - 1) * (q1 - roc_auc ** 2) + (n2 - 1) * (q2 - roc_auc ** 2)) / (n1 * n2)
        )
        print(f"AUC: {round(roc_auc, 2)} +/- {round(auc_err, 2)}")

        # Plot della curva ROC
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Validation ROC')
        plt.legend(loc="lower right")
        plt.show()
        print("[DEBUG] validation_roc: plot mostrato")

    def test_roc(self, x_test, y_test):
        print("[DEBUG] test_roc: inizio")
        confidence_int = 0.683
        z_score = scipy.stats.norm.ppf((1 + confidence_int) / 2.0)

        # Valutazione dell'accuratezza
        test_loss, test_acc = self.evaluate(x_test, y_test, verbose=0)
        print(f"[DEBUG] test_roc: test_acc={test_acc}")
        accuracy_err = z_score * np.sqrt((test_acc * (1 - test_acc)) / y_test.shape[0])
        print(f'Test accuracy: {round(test_acc, 2)} +/- {round(accuracy_err, 2)}')

        # Predizioni e calcolo della curva ROC
        preds_test = self.predict(x_test, verbose=1)
        print(f"[DEBUG] test_roc: preds_test shape {preds_test.shape}, min={preds_test.min()}, max={preds_test.max()}")
        fpr, tpr, _ = roc_curve(y_test, preds_test)
        roc_auc = auc(fpr, tpr)

        # Calcolo dell'errore sull'AUC
        n1 = np.sum(y_test == 1)
        n2 = np.sum(y_test == 0)
        q1 = roc_auc / (2 - roc_auc)
        q2 = 2 * roc_auc ** 2 / (1 + roc_auc)
        auc_err = z_score * np.sqrt(
            (roc_auc * (1 - roc_auc) + (n1 - 1) * (q1 - roc_auc ** 2) + (n2 - 1) * (q2 - roc_auc ** 2)) / (n1 * n2)
        )
        print(f"AUC: {round(roc_auc, 2)} +/- {round(auc_err, 2)}")

        # Plot della curva ROC
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
        Carica un modello salvato, continua l'allenamento e valuta le sue prestazioni.

        Parametri:
        ----------
        path : str
            Percorso del file dei pesi del modello salvato.
        x_train, y_train, x_val, y_val, x_test, y_test : numpy.ndarray
            Dati di addestramento, validazione e test con le rispettive etichette.
        n_epochs : int
            Numero di epoche per il riaddestramento.
        batchsize : int
            Dimensione del batch per il riaddestramento.
        """
        # Compila il modello con i parametri iniziali
        self.compile(optimizer=SGD(learning_rate=0.01), loss=BCE(), metrics=['accuracy'])

        # Esegue un singolo passo di addestramento
        self.train_on_batch(x_train, y_train)

        # Carica i pesi salvati
        self.load_weights(path)

        # Continua l'allenamento e valuta il modello
        self.compile_and_fit(x_train, y_train, x_val, y_val, x_test, y_test, n_epochs, batchsize)
