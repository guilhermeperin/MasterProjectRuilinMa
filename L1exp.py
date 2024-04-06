import h5py
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
attack_set = in_file['Attack_traces/traces'][
             dataset_properties["n_val"]:dataset_properties["n_val"] + dataset_properties["n_attack"]]

# Normalize the data
scaler = StandardScaler()
profiling_set = scaler.fit_transform(profiling_set)
validation_set = scaler.transform(validation_set)
attack_set = scaler.transform(attack_set)

# L1 regularization values to explore
l1_values = np.linspace(0, 2.5e-4, 10)

results = []

def log_details(file_path, details, is_start=False, is_end=False):
    with open(file_path, 'a') as f:
        if is_start:
            f.write("\n--- Start of New Execution ---\n")
        elif is_end:
            f.write("\n--- End of Execution ---\n\n")
        else:
            # Log the details with formatted L1 value
            log_entry = f"{details['timestamp']}, L1: {details['l1']:.3e}, Batch Size: {details['batch_size']}, Epochs: {details['epochs']}, Learning Rate: {details['learning_rate']}, GE: {details['ge']}, NT: {details['nt']}, PI: {details['pi']:.5e}\n"
            f.write(log_entry)


# Loop over L1 regularization values
for i, l1_val in enumerate(l1_values):
    if i == 0:  # Log the start marker only for the first iteration
        log_details('l1_regularization_log.txt', {}, is_start=True)

    print(f"Training with L1 regularization value: {l1_val}")

    # Define the model with L1 regularization
    model = Sequential([
        Dense(100, input_dim=profiling_set.shape[1], activation='elu', kernel_regularizer=l1(l1_val)),
        Dense(100, activation='elu', kernel_regularizer=l1(l1_val)),
        Dense(100, activation='elu', kernel_regularizer=l1(l1_val)),
        Dense(100, activation='elu', kernel_regularizer=l1(l1_val)),
        Dense(classes, activation='softmax')
    ])

    # Compile and train the model
    batch_size = 200  # Dynamic value based on your model's training configuration
    epochs = 10  # Dynamic value based on your model's training configuration
    learning_rate = 0.001  # Dynamic value based on your model's training configuration

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])
    history = model.fit(
        x=profiling_set,
        y=dataset_labels.y_profiling[target_key_byte],
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(validation_set, dataset_labels.y_validation[target_key_byte]),
        shuffle=True
    )

    ge, nt, pi, ge_vector = attack(model, attack_set, dataset_labels, target_key_byte, classes)
    results.append({'L1_value': l1_val, 'GE': ge, 'NT': nt, 'PI': pi})

    # Log the details
    details = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'l1': l1_val,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'ge': ge,
        'nt': nt,
        'pi': pi
    }

    log_details('l1_regularization_log.txt', details)

    if i == len(l1_values) - 1:  # Log the end marker only after the last iteration
        log_details('l1_regularization_log.txt', {}, is_end=True)

    # Clear the session and delete the model
    K.clear_session()
    del model
    gc.collect()  # Explicitly collect garbage

# Plotting NT vs. L1
l1_values = np.array([result['L1_value'] for result in results])
nt_values = np.array([result['NT'] for result in results])

# Sort the values by L1 to ensure correct plotting
sorted_indices = np.argsort(l1_values)
l1_values_sorted = l1_values[sorted_indices]
nt_values_sorted = nt_values[sorted_indices]

# Using LOWESS to smooth the NT values
lowess_smoothed = lowess(nt_values_sorted, l1_values_sorted, frac=0.1)

# Extract unique L1 values and their corresponding min/max NT values
unique_l1_values = np.unique(l1_values_sorted)
min_nt_values = [np.min(nt_values_sorted[l1_values_sorted == l1]) for l1 in unique_l1_values]
max_nt_values = [np.max(nt_values_sorted[l1_values_sorted == l1]) for l1 in unique_l1_values]

plt.figure(figsize=(10, 6))

# Plotting the smoothed trend line
plt.plot(lowess_smoothed[:, 0], lowess_smoothed[:, 1], label='Smoothed NT', color='blue')

# Filling between the min and max boundaries
plt.fill_between(unique_l1_values, min_nt_values, max_nt_values, color='gray', alpha=0.3, label='Min/Max Band')

plt.scatter(l1_values_sorted, nt_values_sorted, color='red', alpha=0.6, label='Original NT')

plt.title('NT vs. L1 Regularization')
plt.xlabel('L1 Regularization Value')
plt.ylabel('NT Metric')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('NT_vs_L1_professional.png')
plt.show()