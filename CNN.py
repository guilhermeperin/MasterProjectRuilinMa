from keras.regularizers import l1
import gc
from keras import backend as K
from datasets import *
from metrics import *
import h5py
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, BatchNormalization, AveragePooling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

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
attack_set = in_file['Attack_traces/traces'][dataset_properties["n_val"]:dataset_properties["n_val"] + dataset_properties["n_attack"]]

# normalize traces
scaler = StandardScaler()
profiling_set = scaler.fit_transform(profiling_set)
validation_set = scaler.transform(validation_set)
attack_set = scaler.transform(attack_set)

# Reshape data for CNN input
profiling_set = profiling_set.reshape(profiling_set.shape[0], profiling_set.shape[1], 1)
validation_set = validation_set.reshape(validation_set.shape[0], validation_set.shape[1], 1)
attack_set = attack_set.reshape(attack_set.shape[0], attack_set.shape[1], 1)

input_size=700
learning_rate=0.0005
l1_val =0.0001

def cnn_architecture_sequential(input_size=700, learning_rate=0.0005, classes=256, l1_val=0.0001):
    model = Sequential([
        # Convolutional block 1
        Conv1D(4, 1, kernel_initializer='he_uniform', activation='selu', padding='same', input_shape=(input_size, 1)),
        BatchNormalization(),
        AveragePooling1D(2, strides=2),
        Flatten(),

        # Dense layers
        Dense(100, activation='selu', kernel_regularizer=l1(l1_val)),
        Dense(100, activation='selu', kernel_regularizer=l1(l1_val)),

        # Output layer
        Dense(classes, activation='softmax')
    ])

    # Compile model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

K.clear_session()
gc.collect()  # Explicitly collect garbage

model = cnn_architecture_sequential()
model.summary()

# Training process with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    x=profiling_set,
    y=dataset_labels.y_profiling[target_key_byte],
    batch_size=200,
    epochs=500,  # Increased epochs, but will rely on early stopping
    verbose=1,
    validation_data=(validation_set, dataset_labels.y_validation[target_key_byte]),
    shuffle=True,
    callbacks=[early_stopping]
)

# metrics
ge, nt, pi, ge_vector = attack(model, attack_set, dataset_labels, target_key_byte, classes)
