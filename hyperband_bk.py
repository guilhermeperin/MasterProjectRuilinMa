import os
from datasets import *
from metrics import *
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import *
from tqdm.keras import TqdmCallback
import sys
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import json
import datetime
import numpy as np
import random
from keras.callbacks import EarlyStopping

from kerastuner import HyperModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import Hyperband


class MLPHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = Sequential()
        model.add(Dense(hp.Int('units_input', min_value=32, max_value=512, step=32),
                        activation='relu', input_shape=(self.input_shape,)))

        # Adding variable number of hidden layers with the same activation function
        for i in range(hp.Int('num_layers', 1, 5)):
            model.add(Dense(hp.Int(f'units_{i}', min_value=32, max_value=512, step=32),
                            activation=hp.Choice('activation', ['relu', 'selu', 'elu'])))

        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(optimizer=Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

current_directory, datasets_path = initialize_path()

# download datasets
# download_ascad_fixed_keys()
# download_ascad_random_keys()
# download_eshard()

# define a target key byte:
target_key_byte = 2
leakage_model = "ID"
if leakage_model == "ID":
    classes = 256
else:
    classes = 9

# labelize and define properties
dataset_labels, dataset_properties = prepare_dataset(datasets_path, "ASCADf", target_key_byte, leakage_model)

# open the samples
in_file = h5py.File(dataset_properties["filepath"], "r")
profiling_set = in_file['Profiling_traces/traces']
validation_set = in_file['Attack_traces/traces'][:dataset_properties["n_val"]]
attack_set = in_file['Attack_traces/traces'][
             dataset_properties["n_val"]:dataset_properties["n_val"] + dataset_properties["n_attack"]]

# normalize traces
scaler = StandardScaler()
profiling_set = scaler.fit_transform(profiling_set)
validation_set = scaler.transform(validation_set)
attack_set = scaler.transform(attack_set)

# define the model
# batch_size = 400
# epochs = 200
# total_epochs = 100
# epochs_per_phase = 20
# activation_functions = ['relu', 'selu', 'sigmoid', 'tanh']


def add_early_stopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=1, mode='auto'):
    early_stopping = EarlyStopping(
        monitor=monitor,
        min_delta=min_delta,
        patience=patience,
        verbose=verbose,
        mode=mode,
        restore_best_weights=True
    )
    return early_stopping


class ElegantProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_file_path):
        super(ElegantProgressCallback, self).__init__()
        self.log_file_path = log_file_path

    def on_train_begin(self, logs=None):
        self.total_epochs = self.params['epochs']

    def on_epoch_end(self, epoch, logs=None):
        # Update progress
        completed = (epoch + 1) / self.total_epochs
        bar_length = 30
        block = int(round(bar_length * completed))
        text = "\rProgress: [{}] {:.0f}% - epoch {}/{} - loss: {:.4f} - accuracy: {:.4f} - val_loss: {:.4f} - val_accuracy: {:.4f}".format(
            "=" * block + " " * (bar_length - block), completed * 100, epoch + 1, self.total_epochs, logs['loss'], logs['accuracy'], logs['val_loss'], logs['val_accuracy'])

        # Log the performance
        with open(self.log_file_path, 'a') as file:
            file.write(f"{text}\n")

    def on_train_end(self, logs=None):
        # Ensure the final log is on a new line
        sys.stdout.write("\n")


# Initialize history
history = {
    'loss': [],
    'accuracy': [],
    'val_loss': [],
    'val_accuracy': []
}

def log_experiment_details(filepath, model, ge, nt, pi):
    with open(filepath, 'a') as file:
        file.write(f"\nExperiment Time: {datetime.datetime.now()}\n")
        file.write("Model Architecture:\n")
        for layer in model.layers:
            layer_details = f"Layer Type: {layer.__class__.__name__}, "
            layer_details += f"Output Units: {layer.output_shape}, "
            layer_details += f"Activation: {layer.activation.__name__ if hasattr(layer, 'activation') else 'N/A'}, "
            layer_details += f"Input Shape: {layer.input_shape if hasattr(layer, 'input_shape') else 'N/A'}\n"
            file.write(layer_details)
        file.write(f"GE: {ge}, Number of Traces to Reach GE: {nt}, PI: {pi}\n")


hypermodel = MLPHyperModel(input_shape=profiling_set.shape[1], num_classes=classes)

tuner = Hyperband(
    hypermodel,
    objective='val_accuracy',
    max_epochs=10,
    directory='hyperband',
    project_name='mlp_tuning'
)

stop_early = EarlyStopping(monitor='val_loss', patience=5)


experiment_log_file = "experiment_log_hp.txt"

tuner.search(profiling_set, dataset_labels.y_profiling[target_key_byte],
             epochs=10,
             validation_data=(validation_set, dataset_labels.y_validation[target_key_byte]),
             callbacks=[ElegantProgressCallback(experiment_log_file), stop_early])

# Retrieve the best model from Hyperband tuner
best_model = tuner.get_best_models(num_models=1)[0]


ge, nt, pi = attack(best_model, validation_set, dataset_labels.y_validation[target_key_byte], target_key_byte, classes)

# Log the best model's architecture and its evaluation metrics
log_experiment_details(experiment_log_file, best_model, ge, nt, pi)

print("\nFinished hyperparameter tuning and model evaluation.")