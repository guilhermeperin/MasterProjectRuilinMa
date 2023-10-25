## Master Project: Efficient Hyperparameter Tuning for Deep Learning-based Profiling SCA (and its relation to different SNR levels in SCA traces)

### Description:

# Cryptographic Algorithms and Side-Channel Analysis

Cryptographic algorithms (e.g., AES, RSA, ECDSA, etc.) are mathematically secure. Knowing the input and output data (i.e., plaintext and ciphertext), as well as every detail about the cryptographic algorithm, is not enough to recover/compute/extract the key. However, when a cryptographic implementation executes on electronic devices (e.g., microprocessors, microcontrollers), there are several sources of unintended and unavoidable information leakages. The most common types are power consumption, electromagnetic emission, execution time, temperature, and acoustics.

An attacker may be able to measure these side-channel information during the execution of encryption or decryption operations and perform statistical analysis to extract the key from the side-channel measurements.

## Side-Channel Analysis (SCA) Research

In the last two decades, side-channel analysis (SCA) research has mainly focused on two common approaches. The first explores direct attacks, in which the main focus is to improve statistical methods, distinguishers, and leakage assessment techniques. Here, we can mention Differential Power Analysis, Correlation Power Analysis, Mutual Information Analysis as the main attack types. The second approach involves the so-called profiling SCA, which follows a similar approach to supervised learning.

In profiling SCA, the side-channel analysis process assumes a threat model in which an adversary possesses an identical copy of the victim's device and is able to collect a set of side-channel measurements from this device. During this collection, the adversary may program different data into the device, such as different cryptographic keys, input/output, and countermeasures-related data. This dataset is considered the training or profiling set. Subsequently, the adversary collects a separate set of side-channel measurements from the target device running an unknown and fixed key. The training data is then used to learn a statistical function or statistical parameters that result in a probability function, usually referred to as a **profiling model**. This function is then applied to the set of measurements collected from the second and victim's device to reveal (the probability of) the key.

Roughly, two main types of profiling models can be built:
1. Those where the probability function is well-defined and require the computation of a few statistical parameters from data (e.g., Gaussian template attacks, in which we compute mean, variance, and covariance).
2. Those models that require statistical parameters to be learned from data (e.g., machine learning techniques).

In recent years, deep neural networks have been massively adopted as profiling models. The usage of deep learning for building profiling models provides several advantages compared to previous techniques, such as Gaussian template attacks. First, side-channel measurements usually contain a large number of sample points per measurement, leading to high-dimensional data. Deep neural networks are capable of handling high-dimensional data and implementing feature selection automatically. Second, side-channel measurements suffer from setup side effects, such as jitter and desynchronization, which can also be intentionally implemented as countermeasures by the cryptographic designer. Convolutional neural networks together with data augmentation techniques provide good robustness against these side effects. Third, the construction of a deep neural network is highly exploratory, leading to infinite possible combinations of hyperparameters. This provides more capacity for deep learning models to learn the side-channel leakages and eventually break countermeasures.

However, the third advantage may also become a curse: finding a good neural network architecture requires expensive hyperparameter tuning efforts. Recent research on deep learning-based side-channel analysis has commonly considered academic targets that are easy to break (even those that are protected with countermeasures), and some light hyperparameter tuning methods are enough to find neural networks (MLPs, CNNs, ResNets, AutoEncoders) with good performances. The most adopted hyperparameter methods are grid search and random search. Reinforcement learning and Bayesian Optimization have been proposed, however with limited publications.

## Main Open Problems

The main open research questions in deep learning-based profiling side-channel analysis, with respect to hyperparameter tuning are:

- How to run a hyperparameter tuning process that increases the convergence to an optimal result (grid and random searches provide no guarantees).
- How to implement a hyperparameter search for different levels of SNR (i.e., noise) in side-channel measurements.
- How to spend hyperparameter tuning resources efficiently (similar to successive halving)? Here we must take into account that aborting a model training too soon just because the metrics indicate bad performance might not be efficient (the model could require a longer training).

## Research Objectives

The main goals of this research are:

- To conduct a literature review of hyperparameter search techniques.
- To propose a hyperparameter tuning method that is more efficient than naive grid and random search processes (i.e., with more guarantees of convergence). There is no need to run random search and grid search, as there are plenty of published papers with results using these methods.
- To propose a method that does self-adaption of its hyperparameter ranges.
- To propose a method in which the hyperparameter search strategy is related to the noise/SNR levels of side-channel measurements.
- Investigate if there is a minimum required SNR level vs. the number of measurements that a hyperparameter tuning can find an efficient method with N searches.

## Datasets

The research can be conducted with the following datasets:

- [ASCAD](https://github.com/ANSSI-FR/ASCAD)
- [ESHARD](https://eshard.com/posts/masked-and-shuffled-dataset-for-sca)
- [DPAv4.2](http://aisylabdatasets.ewi.tudelft.nl/dpav42/)






