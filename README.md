## Master Project: Efficient Hyperparameter Tuning for Deep Learning-based Profiling SCA (and its relation to different SNR levels in SCA traces)

### Description:

Cryptographic algorithms (e.g., AES, RSA, ECDSA, etc) are mathematically secure. Knowing the input and output data (i.e.m plaintext and ciphertext), as well as every detail about the cryptographic 
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

In the last years, deep neural networks have been massively adopted as profiling models. 

