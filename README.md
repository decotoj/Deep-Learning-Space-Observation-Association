# Deep-Learning-Space-Observation-Association

Space Observation Association is the problem of determining which, if any, angles only observations from a group of untagged observations are of the same resident space object (RSO).  Angles only observations refers to optical tracks derived from ground or space based telescropes observing the space environment.  A typical observation consists of the following pieces of information: an epoch, observer position, unit vector to the observed object, and rate of change of the observation unit vector. 

This projects explores methods of solving this problem that do not rely on 'expert system' approaches.  Rather than rely on expert humans to code rulesets this projects attempts to learn from the data itself.

Luckily this is a problem for which realistic data can be readily synthesized.  The file 'synthesize_data.py' can be run to produce simulated tracking data for which the true correlations are known (i.e. labelled training data).  sythesize_data.py produces three separate sets of two files, a train, validation, and test set.  Each set consists of:

# Data CSV File:  
Each row is an observation with the following 10 parameters for each observation.  All parameters are reported in an Earth Centered
Inertial (ECI) coordinate frame.  A simple two body propagator is used to simulate the observations (thank you to Shushi Uetsuki/whiskie14142)

time (seconds)
observer pos x (km)
observer pos y (km)
observer pos z (km)
observation unit vector x 
observation unit vector y 
observation unit vector z 
observation unit vector rate of change x (1/seconds)
observation unit vector rate of change y (1/seconds)
observation unit vector rate of change z (1/seconds)

# Tags CSV File: 
Contains the true RSO id numbers for observed RSOs in the observations in the Data CSV File

'main_nn_assoc_8.py' performs training of a multi-layered fully connected neural network to identify triplets of observation as either all of the same RSO or of one or more RSOs.  The resulting network can be used to identify triplets of observation most likely to be of the same RSO from a large pool of uncorrelated observations.  Such identifcation can then lead to more accurate orbit fits to the unknown object than would have been possible with single track data alone.  The resulting orbit can then be used for sensor re-tasking and ultimately regular observation and custody of the unknown object.

# Data Visualization

Two dimensional display of observations of four different RSOs over a multi-hour period.  Observation association is the task of associating which observations are of the same RSO (w/ no apriori tagging or other information).

![GitHub Logo](/resources/data_example_2D.png)
Format: ![Alt Text](url)

Three dimensional display of several observations.  Observer positions are on surface of earth.  Each observation is a unit vector originating at the location of the observer and in the direction of the observed object.

![GitHub Logo](/resources/data_example_3D.png)
Format: ![Alt Text](url)

