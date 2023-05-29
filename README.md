## Master Project: Efficient Hyperparameter Tuning for Deep Learning-based Profiling SCA (and its relation to different SNR levels in SCA traces)

### Description:

Cryptographic algorithms (e.g., AES, RSA, ECDSA, etc) are mathematically secure. Knowing the input and output data (i.e., plaintext and ciphertext), as well as every detail about the cryptographic 
algorithm, is not enough to recover/compute/extract the key. However, when a cryptographic implementation executes on an electronic devices (e.g., microprocessors, microcontrolers),
there are several sources of unintended and unavoidable information leakages. Most common types are power consumption, electromagnetic emission, execution time, temperature and acoustics.
An attacker may be able to measure these side-channel information during the execution of encryption or decryption operation and to perform statistical analysis to extract the key from 
the side-channel measurements.  

In the last two decades, side-channel analysis (SCA) research have mainly focused on two common approaches. 
The first explores direct attacks, in which the main focus is to improve statistical methods, distinguishers and leakage assessment 
techniques. Here, we can mention Differential Power Analysis, Correlation Power Analysis, Mutual Information Analyis as the main attack
types. The second approach involves the so-calles profiling SCA, which follows a similar approach as supervised learning. In this case, the 
side-channel analysis process assumes a threat model in which an adversary possesses an identical copy of the victim's device and is able to 
collect a set of set side-channel measurements from this device. During this collection, the adversary may program different data to the device,
such as different cryptographic keys, input/output and countermeasures-related data. This dataset is considered as the training or profiling set.
Following, the adversary collects a separate set of side-channel measurements from the target device running an unknown and fixed key. 
The training data is then used to learn a statistical function or statistical parameters that results in a probability function, usually 
referred to as a **profiling model**. This function is then applied to the set of measurements collected from the second and victim's device in order to reval (probability of) the key.

Roughly, two main types of profiling models can be built: (1) those where the probability function is well-defined and requires the computation
of a few statistical parameters from data (e.g., Gaussian template attacks, in which we compute mean, variance and covariance) and (2) those models that 
require that statistical parameters to be learned from data (e.g., machine learning techniques).

In the last years, deep neural networks have been massively adopted as profiling models. For that, a deep neural network is used as profiling model.
The usage of deep learning for building profiling models provides some advantages in comparison to previous techniques such as Gaussian template attacks. 
First, side-channel measurements usually contains a large amount of sample points per measurements, leading to high-dimensional data. Deep neural networks
are able to receive at their inputs high-dimensional data and implement feature selection automatically. Second, side-channel measurements suffer from
setup side effects (which can also be intentionally implemented as a countermeasure by the cryptographic designer) such as jitter and desynchronization. 
Convolutional neural networks together with data augmentation techniques provide good robustness against these side effects. Third, the construction of a deep
neural network is highly exploratory, leading to infinite possible combinations of hyperparameters. This provides more capacity to deep learning models to learn the 
side-channel leakages and eventually break countermeasures. 

This third advantage may also become a curse: finding a good neural network architecture requires expensive hyperparameter tuning efforts. The recent 
research on deep learning-based side-channel analysis have commonly considered academic targets that are easy to break (even those that are protected with countermeasures)
and some light hyperparameter tuning methods are enough to find neural networks (MLPs, CNNs, Resnets, AutoEncoders) with good performances. Most adopted
hyperparameter methods are grid search and random search. Reinforcement learning and Bayesian Optimization have been proposed, however with limited publications.

## Research Objectives







