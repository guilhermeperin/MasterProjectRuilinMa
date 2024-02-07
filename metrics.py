import numpy as np
from scipy.stats import entropy


def guessing_entropy(predictions, labels_guess, good_key, key_rank_attack_traces, key_rank_report_interval=1):
    """
    Function to compute Guessing Entropy
    - this function computes a list of key candidates, ordered by their probability of being the correct key
    - if this function returns final_ge=1, it means that the correct key is actually indicated as the most likely one.
    - if this function returns final_ge=256, it means that the correct key is actually indicated as the least likely one.
    - if this function returns final_ge close to 128, it means that the attack is wrong and the model is simply returing a random key.

    :return
    - final_ge: the guessing entropy of the correct key
    - guessing_entropy: a vector indicating the value 'final_ge' with respect to the number of processed attack measurements
    - number_of_measurements_for_ge_1: the number of processed attack measurements necessary to reach final_ge = 1
    """

    nt = len(predictions)

    key_rank_executions = 40

    # key_ranking_sum = np.zeros(key_rank_attack_traces)
    key_ranking_sum = np.zeros(
        int(key_rank_attack_traces / key_rank_report_interval))

    predictions = np.log(predictions + 1e-36)

    probabilities_kg_all_traces = np.zeros((nt, 256))
    for index in range(nt):
        probabilities_kg_all_traces[index] = predictions[index][
            np.asarray([int(leakage[index])
                        for leakage in labels_guess[:]])
        ]

    for run in range(key_rank_executions):
        r = np.random.choice(
            range(nt), key_rank_attack_traces, replace=False)
        probabilities_kg_all_traces_shuffled = probabilities_kg_all_traces[r]
        key_probabilities = np.zeros(256)

        kr_count = 0
        for index in range(key_rank_attack_traces):

            key_probabilities += probabilities_kg_all_traces_shuffled[index]
            key_probabilities_sorted = np.argsort(key_probabilities)[::-1]

            if (index + 1) % key_rank_report_interval == 0:
                key_ranking_good_key = list(
                    key_probabilities_sorted).index(good_key) + 1
                key_ranking_sum[kr_count] += key_ranking_good_key
                kr_count += 1

    ge = key_ranking_sum / key_rank_executions

    number_of_measurements_for_ge_1 = key_rank_attack_traces
    if ge[0] < 2:
        number_of_measurements_for_ge_1 = 1
    else:
        if ge[int(key_rank_attack_traces / key_rank_report_interval) - 1] < 2:
            for index in range(int(key_rank_attack_traces / key_rank_report_interval) - 1, -1, -1):
                if ge[index] > 2:
                    number_of_measurements_for_ge_1 = (index + 1) * key_rank_report_interval
                    break

    final_ge = ge[int(key_rank_attack_traces / key_rank_report_interval) - 1]
    print("GE = {}".format(final_ge))
    print("Number of traces to reach GE = 1: {}".format(number_of_measurements_for_ge_1))

    return final_ge, ge, number_of_measurements_for_ge_1


def perceived_information(model, labels, num_classes):
    labels = np.array(labels, dtype=np.uint8)
    p_k = np.ones(num_classes, dtype=np.float64)
    for k in range(num_classes):
        p_k[k] = np.count_nonzero(labels == k)
    p_k /= len(labels)

    pi = entropy(p_k, base=2)  # we initialize the value with H(K)

    y_pred = np.array(model + 1e-36)

    for k in range(num_classes):
        trace_index_with_label_k = np.where(labels == k)[0]
        y_pred_k = y_pred[trace_index_with_label_k, k]

        y_pred_k = np.array(y_pred_k)
        if len(y_pred_k) > 0:
            p_k_l = np.sum(np.log2(y_pred_k)) / len(y_pred_k)
            pi += p_k[k] * p_k_l

    print(f"PI: {pi}")

    return pi


def attack(model, attack_traces, dataset_labels, target_key_byte, classes):
    """ Predict the trained MLP with target/attack measurements """
    predictions = model.predict(attack_traces)
    """ Check if we are able to recover the key from the target/attack measurements """
    ge, ge_vector, nt = guessing_entropy(predictions, dataset_labels.labels_key_hypothesis_attack[target_key_byte],
                                         dataset_labels.correct_key_attack[target_key_byte], 2000)
    pi = perceived_information(predictions, dataset_labels.attack_labels[target_key_byte], classes)

    return ge, nt, pi, ge_vector
