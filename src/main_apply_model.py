# Experiment Solving Uncorrelated Space Observation Association Problem
# Last Revision: 11/8/2019
# Authors: XXXX

# Example starter code for how to pass the pre-trained model pairs of observation and
# get back a score of likelihood that each pair is of same satellite.
# Then contains a simple example of applying a threshold to the probability
# to build a policy for each sampled pair of either 0=Match, 1=NoMatch.

import torch
import random
import main_train_model as trnMod
import numpy as np

random.seed(1000)

# Config Parameters
test_data = 'test_data.csv'
test_tags = 'test_tags.csv'
MODEL_FILE = 'model.pt'

# Define Neural Network Layers (matching pretrained model)
model = trnMod.defineModel()
device = torch.device('cpu') #Assign to CPU

# Load Pre-Trained Neural Network Model
model.load_state_dict(torch.load(MODEL_FILE))
model.eval() # Switch Model to Eval Mode

# Parse Test Data
obs, tags = trnMod.loadDataTags(test_data, test_tags)

#######################START: INSERT CODE HERE#################################

# Produce an Example Set of 3 Pairs 
x = [] # list of datapoints to pass to model
z = [] # list of indices for the observation pairs in each datapoint
for k in range(3):
    i = random.randint(0,len(obs)-1)
    j = random.randint(0,len(obs)-1)
    d = [obs[i][:], obs[j][:]] # datapoint consisting of pair of observations
    datapoint = trnMod.augmentAndVectorize(d) 
    x.append(datapoint) 
    z.append([i,j]) 

# Get scores from model for example
x = torch.FloatTensor(x, device=device)
y_ = model(x) #Returns Log Softmax Scores for 0=Match, 1=NoMatch
p = [list(np.exp(q.detach().cpu().numpy()))[0] for q in y_] #Reform scores into probability of match

# Print Results & Build Predictions for Output (example of simple maximum likelihood threshold based method)
output = []
for i in range(len(z)):
    print('Ob Pair Indices: ', z[i], ', Match Score (%): ', p[i], ', True Sat #s: ', tags[z[i][0]], tags[z[i][1]])
    match = 1 #No Match Default
    if p[i] > 0.5:
        match = 0
    output.append([z[i][0], z[i][1], match])

#######################END: INSERT CODE HERE#################################

# Output File for Grading Function
    with open('policy.txt', 'w') as f:
        [f.write(str(q[0]) + ',' + str(q[1]) + ',' + str(q[2]) + '\n') for q in output]