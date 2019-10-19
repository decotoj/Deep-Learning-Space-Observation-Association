# Deep-Learning-Space-Observation-Association

Space Observation Association is the problem of determining which, if any, angles only observations from a group of untagged observations are of the same resident space object (RSO).  Angles only observations refers to optical tracks derived from ground or space based telescropes observing the space environment.  A typical observation consists of the following pieces of information: an epoch, observer position, unit vector to the observed object, and rate of change of the observation unit vector. 

This projects explores methods of solving this problem that do not rely on 'expert system' approaches.  Rather than rely on expert humans to code rulesets this projects attempts to learn from the data itself.

Luckily this is a problem for which realistic data can be readily synthesized.  The file 'synthesize_data.py' can be run to produce simulated tracking data for which the true correlations are known (i.e. labelled training data).

'main_nn_assoc_8.py' performs training of a multi-layered fully connected neural network to identify triplets of observation as either all of the same RSO or of one or more RSOs.  The resulting network can be used to identify triplets of observation most likely to be of the same RSO from a large pool of uncorrelated observations.  Such identifcation can then lead to more accurate orbit fits to the unknown object than would have been possible with single track data alone.  The resulting orbit can then be used for sensor re-tasking and ultimately regular observation and custody of the unknown object.
