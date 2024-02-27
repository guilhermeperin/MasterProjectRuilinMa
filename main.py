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
batch_size = 400
epochs = 200
total_epochs = 80
epochs_per_phase = 10
activation_functions = ['relu', 'selu', 'sigmoid', 'tanh']


def generate_random_configuration():
    num_layers = random.randint(2, 5)  # Choose between 2 and 5 layers
    neurons_per_layer = [random.randint(50, 200) for _ in range(num_layers)]  # Neuron count per layer
    activation = random.choice(activation_functions)  # Choose one activation function for all layers
    return num_layers, neurons_per_layer, activation


def build_model_with_configuration(input_shape, num_classes, num_layers, neurons_per_layer, activation):
    model = Sequential()
    # Input layer
    model.add(Dense(neurons_per_layer[0], input_dim=input_shape, activation=activation))
    # Hidden layers
    for i in range(1, num_layers):
        model.add(Dense(neurons_per_layer[i], activation=activation))
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    return model


# # Define the architecture of the MLP
# model = Sequential()
# model.add(Dense(100, input_dim=profiling_set.shape[1], activation='selu'))
# model.add(Dense(100, activation='selu'))
# model.add(Dense(100, activation='selu'))
# model.add(Dense(100, activation='selu'))
# model.add(Dense(classes, activation='softmax'))

# # Compile the model
# model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])


class ElegantProgressCallback(Callback):
    def on_train_begin(self, logs=None):
        self.total_epochs = self.params['epochs']

    def on_epoch_end(self, epoch, logs=None):
        completed = (epoch + 1) / self.total_epochs
        bar_length = 30
        block = int(round(bar_length * completed))
        text = "\rProgress: [{}] {:.0f}% - epoch {}/{} - loss: {:.4f} - accuracy: {:.4f}".format(
            "=" * block + " " * (bar_length - block), completed * 100, epoch + 1, self.total_epochs, logs['loss'], logs['accuracy'])
        sys.stdout.write(text)
        sys.stdout.flush()

# Initialize history
history = {
    'loss': [],
    'accuracy': [],
    'val_loss': [],
    'val_accuracy': []
}

def save_model_architecture(model, filepath):
    with open(filepath, 'a') as file:
        file.write(f"Experiment Time: {datetime.datetime.now()}\n")
        file.write("\nModel Architecture:\n")
        for layer in model.layers:
            layer_details = f"Layer Type: {layer.__class__.__name__}, "
            layer_details += f"Output Units: {layer.output_shape}, "
            layer_details += f"Activation: {layer.activation.__name__ if hasattr(layer, 'activation') else 'N/A'}, "
            layer_details += f"Input Shape: {layer.input_shape if hasattr(layer, 'input_shape') else 'N/A'}\n"
            file.write(layer_details)

def log_experiment_details(filepath, ge, nt, pi):
    with open(filepath, 'a') as file:
        file.write(f"GE: {ge}, Number of Traces to Reach GE: {nt}, PI: {pi}\n")
        file.write("\n")


experiment_log_file = "experiment_log.txt"

# Train in phases
for phase in range(total_epochs // epochs_per_phase):
    print(f"\nTraining Phase: {phase + 1}/{total_epochs // epochs_per_phase}")

    # Generate a random configuration for this phase
    num_layers, neurons_per_layer, activation = generate_random_configuration()

    # Build the model with the generated configuration
    model = build_model_with_configuration(profiling_set.shape[1], classes, num_layers, neurons_per_layer, activation)

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    # training process with tqdm progress bar
    phase_history = model.fit(
        x=profiling_set,
        y=dataset_labels.y_profiling[target_key_byte],
        batch_size=batch_size,
        epochs=epochs_per_phase,
        verbose=1, # 0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoch
        validation_data=(validation_set, dataset_labels.y_validation[target_key_byte]),
        shuffle=True,
        callbacks=[ElegantProgressCallback()]
    )

    # Append phase history to overall history
    for key in history:
        history[key].extend(phase_history.history[key])

    ge, nt, pi, ge_vector = attack(model, attack_set, dataset_labels, target_key_byte, classes)
    save_model_architecture(model, experiment_log_file)
    log_experiment_details(experiment_log_file, ge, nt, pi)

