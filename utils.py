import h5py
import numpy as np
import os
import shutil
import requests
import zipfile
import estraces
from tqdm import tqdm


def initialize_path():
    current_directory = os.path.dirname(os.path.realpath(__file__))
    datasets_path = os.path.join(current_directory, 'datasets')

    if not os.path.exists(datasets_path):
        os.makedirs(datasets_path)

    return current_directory, datasets_path


def download_file(url, output_file):
    with requests.get(url, stream=True) as response:
        total_size = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size, desc="Downloading", unit="B", unit_scale=True)

        with open(output_file, 'wb') as file:
            for data in response.iter_content(chunk_size=1024):
                file.write(data)
                progress_bar.update(len(data))

        progress_bar.close()


def unzip_file(zip_file_path, extract_to_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_path)


def convert_eshard_to_h5(datasets_filepath, ets_file):
    ths = estraces.read_ths_from_ets_file(ets_file)

    n_profiling = 90000
    n_attack = 10000

    samples = np.array(ths.samples, dtype="float32")
    plaintexts = ths.plaintext
    masks = ths.mask
    keys = ths.key

    profiling_traces = np.array(samples[:n_profiling], dtype="float32")
    attack_traces = np.array(samples[n_profiling:], dtype="float32")

    profiling_plaintexts = plaintexts[:n_profiling]
    profiling_keys = keys[:n_profiling]
    profiling_masks = masks[:n_profiling]

    attack_plaintexts = plaintexts[n_profiling:]
    attack_keys = keys[n_profiling:]
    attack_masks = masks[n_profiling:]

    out_file = h5py.File(os.path.join(datasets_filepath, 'eshard.h5'), 'w')

    profiling_index = [n for n in range(n_profiling)]
    attack_index = [n for n in range(n_attack)]

    profiling_traces_group = out_file.create_group("Profiling_traces")
    attack_traces_group = out_file.create_group("Attack_traces")

    profiling_traces_group.create_dataset(name="traces", data=profiling_traces, dtype=profiling_traces.dtype)
    attack_traces_group.create_dataset(name="traces", data=attack_traces, dtype=attack_traces.dtype)

    metadata_type_profiling = np.dtype([("plaintext", profiling_plaintexts.dtype, (len(profiling_plaintexts[0]),)),
                                        ("key", profiling_keys.dtype, (len(profiling_keys[0]),)),
                                        ("masks", profiling_masks.dtype, (len(profiling_masks[0]),))
                                        ])
    metadata_type_attack = np.dtype([("plaintext", attack_plaintexts.dtype, (len(attack_plaintexts[0]),)),
                                     ("key", attack_keys.dtype, (len(attack_keys[0]),)),
                                     ("masks", attack_masks.dtype, (len(attack_masks[0]),))
                                     ])

    profiling_metadata = np.array(
        [(profiling_plaintexts[n], profiling_keys[n], profiling_masks[n]) for n in profiling_index],
        dtype=metadata_type_profiling)
    profiling_traces_group.create_dataset("metadata", data=profiling_metadata, dtype=metadata_type_profiling)

    attack_metadata = np.array([(attack_plaintexts[n], attack_keys[n], attack_masks[n]) for n in attack_index],
                               dtype=metadata_type_attack)
    attack_traces_group.create_dataset("metadata", data=attack_metadata, dtype=metadata_type_attack)

    out_file.flush()
    out_file.close()
