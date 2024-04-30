import os
import datetime
import h5py
import keras_tuner
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.regularizers import l1
from keras_tuner import HyperModel, tuners
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Custom modules, assuming they are defined elsewhere
from datasets import *
from metrics import *

class MLPHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        # act_func = hp.Choice('activation_function', ['relu', 'elu', 'tanh', 'sigmoid'])
        act_func = hp.Choice('activation_function', ['elu'])
        l1_val = hp.Float('l1_regularization', min_value=0, max_value=2e-4, sampling='linear')

        model = models.Sequential([
            layers.Dense(512, input_dim=self.input_shape, activation=act_func, kernel_regularizer=l1(l1_val)),
            layers.Dense(512, activation=act_func, kernel_regularizer=l1(l1_val)),
            layers.Dense(512, activation=act_func, kernel_regularizer=l1(l1_val)),
            layers.Dense(512, activation=act_func, kernel_regularizer=l1(l1_val)),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        # optimizer_choice = hp.Choice('optimizer', ['adam', 'sgd'])
        optimizer_choice = hp.Choice('optimizer', ['adam'])
        learning_rate = hp.Float('learning_rate', min_value=5e-5, max_value=1.5e-3, sampling='log')

        if optimizer_choice == 'adam':
            optimizer = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_choice == 'sgd':
            optimizer = optimizers.SGD(learning_rate=learning_rate)

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

def log_hyperparameters(log_file, trial_id, hp_values, metrics):
    with open(log_file, 'a') as f:
        f.write(f"{datetime.datetime.now():%y%m%d-%H-%M}, Trial {trial_id}, Hyperparameters: {hp_values}, Metrics: {metrics}\n")

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

def plot_results(results):
    if not results:
        print("No data to plot.")
        return

    # Create the figure and 3D axis
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Extract data from results
    x = [r['learning_rate'] for r in results]
    y = [r['batch_size'] for r in results]
    z = [r['l1_regularization'] for r in results]
    c = [r['NT'] for r in results]  # Color by NT performance values

    # Create the scatter plot
    scatter = ax.scatter(x, y, z, c=c, cmap='viridis', s=60)

    # Customize the axis labels and ticks
    ax.set_xlabel('Learning Rate', fontsize=18, labelpad=20)
    ax.set_ylabel('Batch Size', fontsize=18, labelpad=20)
    ax.set_zlabel('L1 Regularization', fontsize=18, labelpad=35)

    # Customize the tick label size
    ax.tick_params(axis='x', which='major', labelsize=16, pad=10)
    ax.tick_params(axis='y', which='major', labelsize=16, pad=10)
    ax.tick_params(axis='z', which='major', labelsize=16, pad=15)
    ax.set_position([0.1, 0.1, 0.9, 0.9])
    # Set the title of the plot
    fig.suptitle('Impact of Hyperparameters on NT Performance w. Large MLP Model', fontsize=22, x=0.5, y=0.875)

    # Add color bar and customize
    cbar = fig.colorbar(scatter, shrink=0.6, aspect=20, pad=0.15)
    cbar.set_label('NT Performance', rotation=270, labelpad=30, fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H")
    plt.savefig(f'mlp_128_hyperband_{current_time}.png')

    plt.show()

class MyHyperband(tuners.Hyperband):
    def __init__(self, *args, **kwargs):
        super(MyHyperband, self).__init__(*args, **kwargs)
        self.total_epochs_used = 0  # Initialize the epoch counter
    def run_trial(self, trial, *args, **kwargs):
        # Set batch size from the trial's hyperparameters
        epochs = kwargs.get('epochs', 1)  # Defaulting to 1 if not specified
        self.total_epochs_used += epochs
        kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', min_value=32, max_value=1024, step=32)
        # Execute the trial using the super method
        result = super(MyHyperband, self).run_trial(trial, *args, **kwargs)

        model = self.hypermodel.build(trial.hyperparameters)
        checkpoint_dir = os.path.join(self.get_trial_dir(trial.trial_id))
        latest = tf.train.latest_checkpoint(checkpoint_dir)

        if latest is None:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

        model.load_weights(latest).expect_partial()

        # Compute the attack metrics
        ge, nt, pi, ge_vector = attack(model, attack_set, dataset_labels, target_key_byte, classes)

        # Log the trial results
        self.log_trial_details(trial.trial_id, trial.hyperparameters.values, ge, nt, pi)

        # Save the custom metrics for this trial
        self.oracle.update_trial(trial.trial_id, {'GE': ge, 'NT': nt, 'PI': pi, 'score': score(ge, nt)})

        # Collect results for all trials for plotting
        global results
        results.append({
            'learning_rate': trial.hyperparameters.get('learning_rate'),
            'batch_size': trial.hyperparameters.get('batch_size'),
            'l1_regularization': trial.hyperparameters.get('l1_regularization'),
            'NT': nt
        })

        return result
    def log_trial_details(self, trial_id, hp_values, ge, nt, pi):
        with open(log_file, 'a') as f:
            log_entry = f"{datetime.datetime.now():%y%m%d-%H-%M}, Trial {trial_id}, Hyperparameters: {hp_values}, Metrics: GE={ge}, NT={nt}, PI={pi}\n"
            f.write(log_entry)

    def on_search_end(self):
        super(MyHyperband, self).on_search_end()
        with open(log_file, 'a') as f:
            f.write(f"Total epochs used in this run: {self.total_epochs_used}\n")

def score(ge, nt):
    # High penalty for GE not being 1 to ensure models with GE=1 are prioritized
    if ge != 1:
        # Use a high baseline penalty for non-optimal GE, and add a scaled penalty based on the GE value
        return 1000 + ge  # The constant 1000 ensures a high penalty, and adding GE differentiates among non-optimal GEs.
    else:
        # When GE is 1, score solely based on N
        # Normalize NT to a score where lower NT drastically reduces the score
        return nt / 2000  # Here, NT ranges from 1 to 2000, and so the score will range from 0.0005 to 1.

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

results = []
log_file = 'mlp_d512_search_3d_50e_5i_ns_large.txt'
project_name = 'mlp_d512_search_3d_50e_5i_ns_large'

HB_tuner = MyHyperband(
    hypermodel=hypermodel,
    objective=keras_tuner.Objective("score", direction="min"),
    max_epochs=50,
    factor=3,
    hyperband_iterations=5,
    seed=42,
    hyperparameters=None,
    tune_new_entries=True,
    allow_new_entries=True,
    max_retries_per_trial=0,
    max_consecutive_failed_trials=3,
    directory='hyperband',
    project_name=project_name
)

stop_early = callbacks.EarlyStopping(monitor='val_loss', patience=8)
start_time = datetime.datetime.now()

with open(log_file, 'a') as f:
    f.write("--------------------Hyperband_Test_Case_{}--------------------\n".format(project_name))
    f.write("--------------------Hyperband_Trail_Execution_Time_{}--------------------\n".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    f.write("Hyperband_Trail_Execution_Start_Time: {}\n".format(start_time.strftime("%Y%m%d-%H%M%S")))

# Conduct the search
HB_tuner.search(profiling_set, dataset_labels.y_profiling[target_key_byte], epochs=50, validation_data=(validation_set, dataset_labels.y_validation[target_key_byte]),
                callbacks=[stop_early])

end_time = datetime.datetime.now()
duration = end_time - start_time

# After search completes
best_hp = HB_tuner.get_best_hyperparameters()[0]
best_model = HB_tuner.get_best_models(num_models=1)[0]
ge, nt, pi, ge_vector = attack(best_model, attack_set, dataset_labels, target_key_byte, classes)

with open(log_file, 'a') as f:
    f.write("Hyperband_Trail_Execution_End_Time: {}\n".format(end_time.strftime("%Y%m%d-%H%M%S")))
    f.write("Hyperband_Trail_Execution_Duration: {}\n".format(duration))
    f.write("Total epochs used in this run: {}\n".format(HB_tuner.total_epochs_used))

# Logging the best model details
with open(log_file, 'a') as f:
    f.write("\nBest Model Hyperparameters: {}\n".format(best_hp.values))
    f.write("Best Model Performance: GE: {}, NT: {}, PI: {}\n".format(ge, nt, pi))
    f.write("----------------------End of Search Trail----------------------\n")
print('Hyperband search completed.')
plot_results(results)