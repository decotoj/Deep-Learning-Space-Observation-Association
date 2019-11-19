# Experiment Solving Uncorrelated Space Observation Association Problem
# Last Revision: 11/8/2019
# Authors: Jake Decoto

# Utilizes pre-trained model for probability scores and performs Uniform 
# Cost Search to identify triplets of observations that are most likely of
# the same RSO. 

import torch
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
import queue as Q
import numpy as np

# Config Parameters
test_data = 'test_data.csv'
test_tags = 'test_tags.csv'
MODEL_FILE = 'model.pt'
costMax = 0.5 #0.3 #Heuristic maximum cost above which results assumed to not be useful
numObs = 3 # Desired number of observations in each predicted match (e.g. min useful for IOD)
numSol = 200 # Desired number of top solutions returned from UCS algorithm - NOTE: # Unique solutions will be half of this number rounded up 

# Helper Function for Plotting
def bins_labels(bins, **kwargs):
    bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
    plt.xticks(np.arange(min(bins)+bin_w/2, max(bins), bin_w), bins, **kwargs)
    plt.xlim(bins[0], bins[-1])

# Graph for Uniform Cost Search
class Graph:
    def __init__(self, p, k):
        self.edges = {}
        self.weights = {}
        self.p = p
        self.j = k

    def neighbors(self, node):
        i = node[-1]
        neighbors = []
        for j in self.j:
            if j not in node:
                neighbors.append(node + (j,))
        return neighbors

    def get_cost(self, from_node, to_node):
        r = to_node[-1]
        c = 0
        for q in from_node:
            c += (1 - self.p[(q,r)])
            #print('   c', r,q, self.p[(q,r)])
        
        return c

# Unform Cost Search
def ucs(graph, start, goal, numSol):
    visited = set()
    queue = Q.PriorityQueue()
    queue.put((0, start))
    goalStates = []

    while queue:
        cost, node = queue.get()
        if node not in visited:
            visited.add(node)

            if len(node) == goal:
                goalStates.append(node)
                if len(goalStates) >= numSol:
                    return goalStates
            for i in graph.neighbors(node):
                if i not in visited:
                    total_cost = cost + graph.get_cost(node, i)
                    queue.put((total_cost, i))

# Define Neural Network Layers (matching pretrained model)
model = trnMod.defineModel()
device = torch.device('cpu') #Assign to CPU

# Load Pre-Trained Neural Network Model
model.load_state_dict(torch.load(MODEL_FILE))
model.eval() # Switch Model to Eval Mode

# Parse Test Data
obs, tags = trnMod.loadDataTags(test_data, test_tags)
print('# Test Set Obs: ', len(obs))
print('# Unique RSOs: ', len(set(tags)))

# Produce Scores for a Set of Matching and Set of Non-Matching Pairs
c = [] 
rsos = []

for i in range(len(obs)):
#for i in range(119,122):
    p = {} #Dictionary of Scores
    temp = []
    k = []
    for j in range(len(obs)):
        if i != j:
            datapoint = trnMod.augmentAndVectorize([obs[i][:], obs[j][:]]) 
        else:
            datapoint = trnMod.augmentAndVectorize([obs[0][:], obs[-1][:]]) #dummy
        temp.append(datapoint) 

    y = model(torch.FloatTensor(temp, device=device))
    y = [list(np.exp(q.detach().cpu().numpy()))[0] for q in y]
    y[i] = 0
    for q in range(len(y)):
        if 1-y[q] < costMax:
            k.append(q)

    #print('k', len(k))
    if len(k) >= numObs:

        #Add to Dictionary 
        temp = []
        z = []   
        for q in k:
            datapoint = trnMod.augmentAndVectorize([obs[i][:], obs[q][:]]) 
            temp.append(datapoint) 
            z.append((i,q))
            for r in k:
                if q != r:
                    try:
                        tmp = p[(q,r)]
                    except:
                        datapoint = trnMod.augmentAndVectorize([obs[q][:], obs[r][:]]) 
                        temp.append(datapoint) 
                        z.append((q,r))
            if len(z) > 10000:
                y = model(torch.FloatTensor(temp, device=device))
                y = [list(np.exp(q.detach().cpu().numpy()))[0] for q in y]
                for q in range(len(y)):
                    p[z[q]] = y[q] 
                z = []
                temp = []

        if len(temp) > 0:
            y = model(torch.FloatTensor(temp, device=device))
            y = [list(np.exp(q.detach().cpu().numpy()))[0] for q in y]
            for q in range(len(y)):
                p[z[q]] = y[q]         

        scores = []

        graph = Graph(p, k)
        nodes = ucs(graph,(i,), numObs, numSol)
        for node in nodes:
            t = set([tags[q] for q in node])
            #print('node',i, node, len(t))
            scores.append(len(t))
            if len(t) == 1:
                rsos.append(tags[node[0]])

        c.append(min(scores))
        print(i, c[-1], len(set(rsos)))

    else:
        c.append(numObs)
        print('not enough')

# Print Summary of Results
print('1 right count', c.count(1))
print('2 right count', c.count(2))
print('3 right count', c.count(3))
print('# Unique RSOs Acquired: ', len(set(rsos)))
print('# IOD Runs: ', numSol/2*len(obs))

# Plot
fig, ax = plt.subplots()
ax.grid()
ax.set(xlabel='Truth Number of RSOs in Triplet', ylabel='# Occurances',title='UCS Observation Triplet Association Results')
bins = [1,2,3,4]
sns.distplot(c, bins=bins, kde=False)
bins_labels(bins, fontsize=20)
fig.savefig("triplet_results.png")
plt.draw()
plt.show()