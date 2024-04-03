import h5py
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l1
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from datasets import *
from metrics import *
import matplotlib.pyplot as plt
from datetime import datetime

# Initialize paths and download datasets
current_directory, datasets_path = initialize_path()
# download_ascad_fixed_keys()
# download_ascad_random_keys()
# download_eshard()

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
l1_values = np.linspace(0, 1e-4, 10)

# Results storage
results = []

def log_details(file_path, details):
    with open(file_path, 'a') as f:
        # Check if the file is empty to add the header
        f.seek(0, 2)  # Move the cursor to the end of the file
        if f.tell() == 0:  # If file is empty, write the header
            header = "Timestamp, L1 Regularization, Batch Size, Epochs, Learning Rate, GE, NT, PI\n"
            f.write(header)

        # Log the details
        log_entry = f"{details['timestamp']}, {details['l1']}, {details['batch_size']}, {details['epochs']}, {details['learning_rate']}, {details['ge']}, {details['nt']}, {details['pi']}\n"
        f.write(log_entry)


# Loop over L1 regularization values
for l1_val in l1_values:
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
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    history = model.fit(
        x=profiling_set,
        y=dataset_labels.y_profiling[target_key_byte],
        batch_size=100,
        epochs=10,
        verbose=1,
        validation_data=(validation_set, dataset_labels.y_validation[target_key_byte]),
        shuffle=True
    )

    # Evaluate the model
    ge, nt, pi, ge_vector = attack(model, attack_set, dataset_labels, target_key_byte, classes)

    # Store results
    results.append({'L1_value': l1_val, 'GE': ge, 'NT': nt, 'PI': pi})

    # Log the details
    # Usage of the log_details function within your loop
    details = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'l1': l1_val,
        'batch_size': 200,
        'epochs': 8,
        'learning_rate': 0.001,
        'ge': ge,
        'nt': nt,
        'pi': pi
    }

    log_details('l1_regularization_log.txt', details)

# Plotting NT vs. L1
nt_values = [result['NT'] for result in results]
plt.figure(figsize=(10, 6))
plt.plot(l1_values, nt_values, marker='o', linestyle='-', color='b')
plt.title('NT vs. L1 Regularization')
plt.xlabel('L1 Regularization Value')
plt.ylabel('NT Metric')
plt.grid(True)
plt.savefig('NT_vs_L1.png')
plt.show()
