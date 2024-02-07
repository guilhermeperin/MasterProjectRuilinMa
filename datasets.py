from utils import *
from labelize import *


def download_ascad_fixed_keys():
    current_directory, datasets_path = initialize_path()

    url_ascadf = "https://www.data.gouv.fr/s/resources/ascad/20180530-163000/ASCAD_data.zip"
    ascadf_dataset_zip = "ASCAD_data.zip"

    if not os.path.exists(os.path.join(datasets_path, "ASCAD.h5")):
        if not os.path.exists(os.path.join(datasets_path, ascadf_dataset_zip)):
            download_file(url_ascadf, ascadf_dataset_zip)

            file_path_dataset = os.path.join(current_directory, ascadf_dataset_zip)
            destination_directory = 'datasets'
            shutil.move(file_path_dataset, destination_directory)

        if os.path.exists(os.path.join(datasets_path, ascadf_dataset_zip)):

            if not os.path.exists(os.path.join(datasets_path, "ASCAD_data\\ASCAD_databases\\ASCAD.h5")):
                zip_file_path = os.path.join(datasets_path, ascadf_dataset_zip)
                unzip_file(zip_file_path, datasets_path)

            if not os.path.exists(os.path.join(datasets_path, "ASCAD.h5")):
                src_file = os.path.join(datasets_path, "ASCAD_data\\ASCAD_databases\\ASCAD.h5")
                shutil.move(src_file, datasets_path)


def download_ascad_random_keys():
    current_directory, datasets_path = initialize_path()

    url_ascadr = "https://static.data.gouv.fr/resources/ascad-atmega-8515-variable-key/20190903-083349/ascad-variable.h5"
    ascadr_dataset = "ascad-variable.h5"

    if not os.path.exists(os.path.join(datasets_path, ascadr_dataset)):
        download_file(url_ascadr, ascadr_dataset)

        file_path_dataset = os.path.join(current_directory, ascadr_dataset)
        destination_directory = 'datasets'
        shutil.move(file_path_dataset, destination_directory)


def download_eshard():
    current_directory, datasets_path = initialize_path()

    url_eshard = "https://gitlab.com/eshard/nucleo_sw_aes_masked_shuffled/-/raw/main/Nucleo_AES_masked_non_shuffled.ets"
    eshard_dataset_ets = "eshard.ets"
    eshard_dataset = "eshard.h5"

    if not os.path.exists(os.path.join(datasets_path, eshard_dataset_ets)):
        download_file(url_eshard, eshard_dataset_ets)

        file_path_dataset = os.path.join(current_directory, eshard_dataset_ets)
        destination_directory = 'datasets'
        shutil.move(file_path_dataset, destination_directory)

    if not os.path.exists(os.path.join(datasets_path, eshard_dataset)):
        convert_eshard_to_h5(datasets_path, os.path.join(datasets_path, eshard_dataset_ets))


def prepare_dataset(datasets_directory, dataset_select, target_byte, leakage_model):
    if dataset_select == "ASCADf":
        dataset_name = "ascadf"
        dataset_filepath = os.path.join(datasets_directory, "ASCAD.h5")
        n_prof = 50000
        n_val = 5000
        n_attack = 5000
        ns = 10000
    elif dataset_select == "ASCADr":
        dataset_name = "ascadr"
        dataset_filepath = os.path.join(datasets_directory, "ascad-variable.h5")
        n_prof = 200000
        n_val = 5000
        n_attack = 5000
        ns = 25000
    else:
        dataset_name = "ESHARD"
        dataset_filepath = os.path.join(datasets_directory, "eshard.h5")
        n_prof = 90000
        n_val = 5000
        n_attack = 5000
        ns = 1400

    dataset_labels = TargetLabels(dataset_name, n_prof, n_val, n_attack, target_byte, leakage_model, dataset_filepath)

    dataset_properties = {
        "name": dataset_name,
        "filepath": dataset_filepath,
        "n_prof": n_prof,
        "n_val": n_val,
        "n_attack": n_attack,
        "ns": ns
    }

    return dataset_labels, dataset_properties


download_ascad_fixed_keys()
download_ascad_random_keys()
download_eshard()
