import h5py
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l1
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from datasets import *
from metrics import *
from datetime import datetime
import gc
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from statsmodels.nonparametric.smoothers_lowess import lowess

# Initialize paths and download datasets
current_directory, datasets_path = initialize_path()
download_ascad_fixed_keys()
download_ascad_random_keys()
download_eshard()

# Define a target key byte and leakage model
target_key_byte = 2
leakage_model = "ID"
classes = 256 if leakage_model == "ID" else 9

# Prepare the dataset
dataset_labels, dataset_properties = prepare_dataset(datasets_path, "ASCADf", target_key_byte, leakage_model)

# Load the data
in_file = h5py.File(dataset_properties["filepath"], "r")
profiling_set = in_file['Profiling_traces/traces']
validation_set = in_file['Attack_traces/traces'][:dataset_properties["n_val"]]
attack_set = in_file['Attack_traces/traces'][dataset_properties["n_val"]:dataset_properties["n_val"] + dataset_properties["n_attack"]]

# Normalize the data
scaler = StandardScaler()
profiling_set = scaler.fit_transform(profiling_set)
validation_set = scaler.transform(validation_set)
attack_set = scaler.transform(attack_set)

# Batch sizes to explore
batch_sizes = np.arange(8, 1536, 8)  # Batch sizes from 16 to 512, inclusive, in steps of 16

results = []
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

def log_details(file_path, details, is_start=False, is_end=False):
    with open(file_path, 'a') as f:
        if is_start:
            f.write("\n--- Start of New Execution ---\n")
        elif is_end:
            f.write("\n--- End of Execution ---\n\n")
        else:
            log_entry = f"{details['timestamp']}, Batch Size: {details['batch_size']}, Run: {details['run']}, Epochs: {details['epochs']}, L1: {details['l1']:.3e}, Learning Rate: {details['learning_rate']:.3e}, GE: {details['ge']}, NT: {details['nt']}, PI: {details['pi']:.5e}\n"
            f.write(log_entry)

# Loop over batch size values
for batch_size in batch_sizes:
    for run in range(1, 6):  # Perform 5 runs for each batch size
        print(f"Training with batch size: {batch_size}, Run: {run}")

        model = Sequential([
            Dense(100, input_dim=profiling_set.shape[1], activation='elu', kernel_regularizer=l1(1e-4)),
            Dense(100, activation='elu', kernel_regularizer=l1(1e-4)),
            Dense(100, activation='elu', kernel_regularizer=l1(1e-4)),
            Dense(100, activation='elu', kernel_regularizer=l1(1e-4)),
            Dense(classes, activation='softmax')
        ])

        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=5e-4), metrics=['accuracy'])
        history = model.fit(
            x=profiling_set,
            y=dataset_labels.y_profiling[target_key_byte],
            batch_size=batch_size,
            epochs=10,
            verbose=1,
            validation_data=(validation_set, dataset_labels.y_validation[target_key_byte]),
            shuffle=True,
            callbacks=[early_stopping]
        )

        ge, nt, pi, ge_vector = attack(model, attack_set, dataset_labels, target_key_byte, classes)
        results.append({'batch_size': batch_size, 'run': run, 'GE': ge, 'NT': nt, 'PI': pi})

        details = {
            'timestamp': datetime.now().strftime("%Y-%m-%d_%H"),
            'batch_size': batch_size,
            'run': run,
            'epochs': 10,
            'l1': 1e-4,
            'learning_rate': 5e-4,
            'ge': ge,
            'nt': nt,
            'pi': pi
        }

        log_details('batch_size_log.txt', details)

        K.clear_session()
        del model
        gc.collect()

# Plotting NT vs. Batch Size with min/max band
# Prepare data for plotting
all_batch_sizes = [result['batch_size'] for result in results]
all_nt_values = [result['NT'] for result in results]

# Using LOWESS to smooth the NT values
lowess_smoothed = lowess(all_nt_values, all_batch_sizes, frac=0.1, it=0)

# Sort batch sizes and NT values for min/max band calculation
unique_batch_sizes = sorted(set(all_batch_sizes))
min_nt_values = [min([result['NT'] for result in results if result['batch_size'] == bs]) for bs in unique_batch_sizes]
max_nt_values = [max([result['NT'] for result in results if result['batch_size'] == bs]) for bs in unique_batch_sizes]

plt.figure(figsize=(10, 6))

# Plotting the smoothed trend line
plt.plot(lowess_smoothed[:, 0], lowess_smoothed[:, 1], label='Smoothed NT', color='blue')

# Filling between the min and max boundaries
plt.fill_between(unique_batch_sizes, min_nt_values, max_nt_values, color='gray', alpha=0.3, label='Min/Max NT Band')

# Plotting original NT points
plt.scatter(all_batch_sizes, all_nt_values, color='red', alpha=0.6, label='Original NT')

plt.title('NT vs. Batch Size')
plt.xlabel('Batch Size')
plt.ylabel('NT Metric')
plt.legend()
plt.grid(True)
plt.tight_layout()
current_time = datetime.now().strftime("%Y-%m-%d_%H")
plt.savefig(f'NT_vs_Batch_Size_{current_time}.png')
plt.show()
