import os
import datetime
import h5py
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.regularizers import l1
from keras_tuner import HyperModel, tuners
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Custom modules, assuming they are defined elsewhere
from datasets import *
from metrics import *

class MLPHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        act_func = hp.Choice('activation_function', ['relu', 'elu', 'tanh', 'sigmoid'])
        l1_val = hp.Float('l1_regularization', min_value=0, max_value=1.5e-4, sampling='linear')

        model = models.Sequential([
            layers.Dense(256, input_dim=self.input_shape, activation=act_func, kernel_regularizer=l1(l1_val)),
            layers.Dense(256, activation=act_func, kernel_regularizer=l1(l1_val)),
            layers.Dense(256, activation=act_func, kernel_regularizer=l1(l1_val)),
            layers.Dense(256, activation=act_func, kernel_regularizer=l1(l1_val)),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        optimizer_choice = hp.Choice('optimizer', ['adam', 'sgd'])
        learning_rate = hp.Float('learning_rate', min_value=5e-5, max_value=1e-3, sampling='log')

        if optimizer_choice == 'adam':
            optimizer = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_choice == 'sgd':
            optimizer = optimizers.SGD(learning_rate=learning_rate)

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

def log_hyperparameters(log_file, trial_id, hp_values, metrics):
    with open(log_file, 'a') as f:
        f.write(f"{datetime.datetime.now():%y%m%d-%H-%M}, Trial {trial_id}, Hyperparameters: {hp_values}, Metrics: {metrics}\n")

def plot_results(results):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = [r['learning_rate'] for r in results]
    y = [r['batch_size'] for r in results]
    z = [r['l1_regularization'] for r in results]
    c = [r['NT'] for r in results]  # NT as the color dimension

    img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
    fig.colorbar(img)
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Batch Size')
    ax.set_zlabel('L1 Regularization')
    plt.title('Impact of Hyperparameters on NT Performance')
    plt.show()

class MyHyperband(tuners.Hyperband):
    def run_trial(self, trial, *args, **kwargs):
        # Customize the batch size using the hyperparameters defined in the trial
        kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', min_value=64, max_value=512, step=16)
        # It's important to capture the output of super().run_trial and return it
        return super(MyHyperband, self).run_trial(trial, *args, **kwargs)


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


hypermodel = MLPHyperModel(input_shape=profiling_set.shape[1], num_classes=classes)
HB_tuner = MyHyperband(
    hypermodel=hypermodel,
    objective='val_accuracy',
    max_epochs=30,
    factor=3,
    hyperband_iterations=3,
    seed=42,
    hyperparameters=None,
    tune_new_entries=True,
    allow_new_entries=True,
    max_retries_per_trial=0,
    max_consecutive_failed_trials=3,
    directory='hyperband',
    project_name='mlp_tuning_7'
)

results = []
log_file = 'mlp_256_hyperband_log.txt'

# Logging the start of Hyperband tests
with open(log_file, 'a') as f:
    f.write("--------------------hyperband_test_mlp_256_{}--------------------\n".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

stop_early = callbacks.EarlyStopping(monitor='val_loss', patience=5)
HB_tuner.search(profiling_set, dataset_labels.y_profiling[target_key_byte], epochs=30, validation_data=(validation_set, dataset_labels.y_validation[target_key_byte]),
                callbacks=[stop_early])

# After search completes
best_hp = HB_tuner.get_best_hyperparameters()[0]
best_model = HB_tuner.get_best_models(num_models=1)[0]
ge, nt, pi, ge_vector = attack(best_model, validation_set, dataset_labels, target_key_byte, classes)

# Logging the best model details
with open(log_file, 'a') as f:
    f.write("\nBest Model Hyperparameters: {}\n".format(best_hp.values))
    f.write("Best Model Performance: GE: {}, NT: {}, PI: {}\n".format(ge, nt, pi))
    f.write("----------------------end search trail----------------------\n")

# Assuming the function `attack` returns a dictionary of metrics for each trial
for trial in HB_tuner.oracle.get_best_trials(num_trials=10):
    result = {'learning_rate': trial.hyperparameters.get('learning_rate'), 'batch_size': trial.hyperparameters.get('batch_size'),
              'l1_regularization': trial.hyperparameters.get('l1_regularization'), 'NT': trial.metrics.get_best_value('NT')}
    results.append(result)

plot_results(results)
