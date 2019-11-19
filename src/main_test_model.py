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
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

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

print('# Test Set Obs: ', len(obs))

#######################START: INSERT CODE HERE#################################

# Produce Scores for a Set of Matching and Set of Non-Matching Pairs
x0 = [] # list of datapoints to pass to model
p0 = []
x1 = [] # list of datapoints to pass to model
p1 = []
for i in range(len(obs)):
    for j in range(len(obs)):

        if i != j:
            d = [obs[i][:], obs[j][:]] # datapoint consisting of pair of observations
            datapoint = trnMod.augmentAndVectorize(d) 
            if tags[i] != tags[j]:
                x0.append(datapoint) 
            else:
                x1.append(datapoint) 

    print('i', i)
    if len(x0) > 0:
        y0 = model(torch.FloatTensor(x0, device=device))
        x0 = []
        p0tmp = [list(np.exp(q.detach().cpu().numpy()))[0] for q in y0]
        p0 += p0tmp
    if len(x1) > 0:
        y1 = model(torch.FloatTensor(x1, device=device))
        x1 = []
        p1tmp = [list(np.exp(q.detach().cpu().numpy()))[0] for q in y1]
        p1 += p1tmp

print('# Non-Match: ', len(p0), 'Mean Score Non-Matches: ', sum(p0)/len(p0))
print('# Non-Match: ', len(p1), 'Mean Score Non-Matches: ', sum(p1)/len(p1))

#Plot Distribution
k = 10
c = [q/k for q in range(0,k+1)]
y0 = [0]*(len(c))
for p in p0:
    y0[int(round(p*k,0))] += 1
y1 = [0]*(len(c))
for p in p1:
    y1[int(round(p*k,0))] += 1

y2 = [y1[i]/(y1[i]+y0[i]) for i in range(len(y0))]

y0 = [q/sum(y0) for q in y0]
y1 = [q/sum(y1) for q in y1]

fig, ax = plt.subplots()
# ax.plot(c, y0)
# ax.plot(c, y1)
ax.plot(c, y2, linewidth=6)
ax.set(xlabel='Model Score', ylabel='Probability of Match',title='Model Applied to Sparsely Matched Test Data')
ax.grid()

text = 'Observations (n): ' + str(int(len(obs))) + '\n' + 'Pair Combinations: 4.995e5\n' + 'Probability of Pair Match: ' + str(round(len(p1)/(len(p1)+len(p0)),4))
at = AnchoredText(text,prop=dict(size=12), frameon=True,loc='upper left')
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax.add_artist(at)

fig.savefig("prob_match_vs_score.png")
plt.draw()
plt.show()

fig, ax = plt.subplots()
ax.plot(c, y0, linewidth=6, label='non-matches')
ax.plot(c, y1, linewidth=6, label='matches')
ax.set(xlabel='Model Score', ylabel='Distribution',title='Distribution of Test Data Pairs')
ax.grid()

text = 'Observations (n): ' + str(int(len(obs))) + '\n' + 'Pair Combinations: 4.995e5\n' + 'Matches: ' + str(len(p1)/2) + '\n' + 'Non-Matches: ' + str(len(p0)/2)
at = AnchoredText(text,prop=dict(size=12), frameon=True,loc='center')
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax.add_artist(at)

ax.legend()

fig.savefig("prob_distribution_vs_score.png")
plt.draw()
plt.show()


