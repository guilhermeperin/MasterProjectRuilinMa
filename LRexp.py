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

# Learning rate values to explore
learning_rates = np.logspace(np.log10(5e-5), np.log10(5e-3), 200)

results = []
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

def log_details(file_path, details, is_start=False, is_end=False):
    with open(file_path, 'a') as f:
        if is_start:
            f.write("\n--- Start of New Execution ---\n")
        elif is_end:
            f.write("\n--- End of Execution ---\n\n")
        else:
            # Log the details with formatted learning rate value
            log_entry = f"{details['timestamp']}, Learning Rate: {details['learning_rate']:.3e}, Batch Size: {details['batch_size']}, Epochs: {details['epochs']}, L1: {details['l1']}, GE: {details['ge']}, NT: {details['nt']}, PI: {details['pi']:.5e}\n"
            f.write(log_entry)

# Loop over learning rate values
for i, lr in enumerate(learning_rates):
    if i == 0:  # Log the start marker only for the first iteration
        log_details('learning_rate_log.txt', {}, is_start=True)

    print(f"Training with learning rate: {lr}")

    # Define the model (with L2 regularization as before)
    model = Sequential([
        Dense(100, input_dim=profiling_set.shape[1], activation='elu', kernel_regularizer=l1(1e-4)),
        Dense(100, activation='elu', kernel_regularizer=l1(1e-4)),
        Dense(100, activation='elu', kernel_regularizer=l1(1e-4)),
        Dense(100, activation='elu', kernel_regularizer=l1(1e-4)),
        Dense(classes, activation='softmax')
    ])

    batch_size = 200
    epochs = 10
    l1_value = 0.0001

    # Compile and train the model with the current learning rate
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
    history = model.fit(
        x=profiling_set,
        y=dataset_labels.y_profiling[target_key_byte],
        batch_size=200,
        epochs=10,
        verbose=1,
        validation_data=(validation_set, dataset_labels.y_validation[target_key_byte]),
        shuffle=True,
        callbacks=[early_stopping]
    )

    ge, nt, pi, ge_vector = attack(model, attack_set, dataset_labels, target_key_byte, classes)
    results.append({'learning_rate': lr, 'GE': ge, 'NT': nt, 'PI': pi})

    # Log the details
    details = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'learning_rate': lr,
        'batch_size': batch_size,
        'epochs': epochs,
        'l1': l1_value,
        'ge': ge,
        'nt': nt,
        'pi': pi
    }

    log_details('learning_rate_log.txt', details)

    if i == len(learning_rates) - 1:  # Log the end marker only after the last iteration
        log_details('learning_rate_log.txt', {}, is_end=True)

    # Clear the session and delete the model
    K.clear_session()
    del model
    gc.collect()  # Explicitly collect garbage

# Plotting NT vs. Learning Rate with min/max band
learning_rates = np.array([result['learning_rate'] for result in results])
nt_values = np.array([result['NT'] for result in results])

# Sort the values by learning rate to ensure correct plotting
sorted_indices = np.argsort(learning_rates)
learning_rates_sorted = learning_rates[sorted_indices]
nt_values_sorted = nt_values[sorted_indices]

# Using LOWESS to smooth the NT values
lowess_smoothed = lowess(nt_values_sorted, learning_rates_sorted, frac=0.1)

# Extract unique learning rate values and their corresponding min/max NT values
unique_learning_rates = np.unique(learning_rates_sorted)
min_nt_values = [np.min(nt_values_sorted[learning_rates_sorted == lr]) for lr in unique_learning_rates]
max_nt_values = [np.max(nt_values_sorted[learning_rates_sorted == lr]) for lr in unique_learning_rates]
current_time = datetime.now().strftime("%Y-%m-%d_%H")
plt.figure(figsize=(10, 6))

# Plotting the smoothed trend line
plt.plot(lowess_smoothed[:, 0], lowess_smoothed[:, 1], label='Smoothed NT', color='blue')

# Filling between the min and max boundaries
plt.fill_between(unique_learning_rates, min_nt_values, max_nt_values, color='gray', alpha=0.3, label='Min/Max Band')

plt.scatter(learning_rates_sorted, nt_values_sorted, color='red', alpha=0.6, label='Original NT')

plt.title('NT vs. Learning Rate')
plt.xscale('log')  # Learning rates are on a log scale
plt.xlabel('Learning Rate')
plt.ylabel('NT Metric')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'NT_vs_Learning_Rate_{current_time}.png')
plt.show()
