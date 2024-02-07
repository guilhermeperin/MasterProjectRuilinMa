from datasets import *
from metrics import *
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import *

current_directory, datasets_path = initialize_path()

# download datasets
download_ascad_fixed_keys()
download_ascad_random_keys()
download_eshard()

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
epochs = 20

# Define the architecture of the MLP
model = Sequential()
model.add(Dense(100, input_dim=profiling_set.shape[1], activation='elu'))
model.add(Dense(100, activation='elu'))
model.add(Dense(100, activation='elu'))
model.add(Dense(100, activation='elu'))
model.add(Dense(classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# training process
history = model.fit(
    x=profiling_set,
    y=dataset_labels.y_profiling[target_key_byte],
    batch_size=batch_size,
    epochs=epochs,
    verbose=2,
    validation_data=(validation_set, dataset_labels.y_validation[target_key_byte]),
    shuffle=True,
    callbacks=[]
)

# metrics
ge, nt, pi, ge_vector = attack(model, attack_set, dataset_labels, target_key_byte, classes)
