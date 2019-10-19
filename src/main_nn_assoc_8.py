# Experiment Training a Neural Network to Aid Optical Space Observation Association
# Last Revision: 10/19/2019
# Authors: Jake Decoto, XXXX

# Next Steps
#1) Audit code (main and sythesize) and hone down to minimum
#2) Make 100% Sure Something is Being Learned (Generalizes)
#3) Validate two body propagator used in synthesize vs external source

# The following loads prebuilt training and test data consisting of optical observerations
# of satellites.  A neural network is then trained to identify pairs of optical observations
# in the labelled training data as being of the same satellite or not.  This network is then
# used to identify triplets of observations in the test data that are most likely to be of 
# the same satellite.  No apriori knowledge of physics or orbital mechanics is used.

#Next Steps:
#A) Reduce false positive triplets, 1) loss function for doubles penalize false+, 2) separate triplet NN for those identified by doublet NN
#B) Separate out grading and knowledge of tags into separate callable function (so can be run as unknowing participant)
#C) Revisit simulated data settings to ensure data is sufficiently real world problem representive
#D) Explore possible performance

import torch
import random
import numpy as np
import torch.optim as optim
from collections import defaultdict

random.seed(100)

#Network and Data Parameters
LOAD_SAVED_MODEL_FLAG = 1 #1=Load prior saved model, 0=Train model from scratch
TRAIN_FLAG = 1 #1=TRAIN, 0=EVAL ONLY
MODEL_FILE_IN = 'model_save.pt' #Path to saved model file if applicable
MODEL_FILE_OUT = 'model_save_update.pt' #Path to saved new model to
D_in = 416 #Length of Input Vectors after Augmentation (time, observer x-y-z unit vec, observervation x-y-z unit vec, streak direction x-y-z unit vec, observer vector mag)
H = 416 #Hidden layer(s) dimension
N = 1000#40000 #Number of training examples to create from loaded data (40000 is most done)
Nval = 1000#2500 #Number of validation examples to create from loaded data (2500 is most done w/ 2 val sets)
NUMEPOCHS = 200
learning_rate = 1e-4
train_data = 'train_data.csv'
train_tags = 'train_tags.csv'
val_data = 'train_data.csv'
val_tags = 'train_tags.csv'
test_data = 'test_data.csv'
test_tags = 'test_tags.csv'

device = torch.device('cpu') #Assign to CPU

def augmentAndVectorize(ob):

    #Sort Observations by Epoch and Normalize Epoch
    ob.sort(key=lambda x: x[0])
    for q in range(len(ob)):
        ob[q][0] = ob[q][0] - ob[0][0]

    #Add Derived Parameter from Observation Unit Vector and Streak Vector and Magnitude (416)
    n = len(ob[0])
    for z in range(1, len(ob)):
        for q in [4,5,6,7,8,9,10]:
            for r in [4,5,6,7,8,9,10]:
                for s in [0,4,5,6,7,8,9,10]:
                    ob[z].append((ob[z][q]-ob[z-1][r])/(ob[z][s]-ob[z-1][s]))

    #Vectorize
    datapoint = []
    for q in range(len(ob)):
         datapoint = datapoint + ob[q] 

    return datapoint 

#Build Labelled Dataset of Observation Doubles
def buildLabelledDataset(n, obs, tags):
    x = []
    y = []
    usedTags = []
    ln = len(obs[0])
    for i in range(n):

        #Find Matching Double
        while True:
            n = random.randint(0, len(tags)-1) 
            indices = [i for i, x in enumerate(tags) if x == tags[n]]
            if len(indices) > 1:
                d = [obs[indices[0]][0:ln], obs[indices[1]][0:ln]]
                x.append(augmentAndVectorize(d) )
                y.append(0)
                usedTags.append(tags[n])
                break

        #Find Non-Matching Double
        while True:
            n2 = random.randint(0, len(tags)-1) 
            if tags[n] != tags[n2]:
                d = [obs[n][0:ln], obs[n2][0:ln]] 
                x.append(augmentAndVectorize(d))
                y.append(1)
                usedTags.append(tags[n2])
                break

    return x, y
    
#Pull Training Data and Tags from input files
def loadDataTags(file1, file2):

    with open(file1, 'r') as f:
        obs = [q.split(',') for q in f.readlines()]
    for i in range(len(obs)):
        obs[i] = [float(q) for q in obs[i]]
        obs[i].append(np.linalg.norm(obs[i][-3:]))#Add Streak Length

        #Refactor Observer Position Into Unit Vector and Magnitude
        R = np.linalg.norm(obs[i][1:4])
        obs[i][1] = obs[i][1] /R
        obs[i][2] = obs[i][2] /R
        obs[i][3] = obs[i][3] /R
        obs[i].append(R)

    if file2 != 'NA':
        with open(file2, 'r') as f:
            tags = [int(q) for q in f.readlines()]
        
    return obs, tags

if TRAIN_FLAG == 1:

    obs, tags = loadDataTags(train_data, train_tags) #Load Training Data
    obs_val, tags_val = loadDataTags(val_data, val_tags) #Load Validation Data
    obs_val2, tags_val2 = loadDataTags(val_data, val_tags) #Load Validation Data

    #Build Training Dataset
    x, y = buildLabelledDataset(N, obs, tags)
    print('Training Data & # RSOs Used', len(x), len(y))

    #Build Validation Dataset
    x_val, y_val = buildLabelledDataset(Nval, obs_val, tags_val)
    print('Validation Data & # RSOs Used',  len(x_val), len(y_val))

    #Build Validation Dataset
    x_val2, y_val2 = buildLabelledDataset(Nval, obs_val2, tags_val2)
    print('Validation Data 2 & # RSOs Used',  len(x_val2), len(y_val2))

    print('/n Total RSOs', len(set(tags)))

    #Assign Training and Val Data
    x = torch.FloatTensor(x, device=device)
    y = torch.LongTensor(y, device=device)
    x_val = torch.FloatTensor(x_val, device=device)
    y_val = torch.LongTensor(y_val, device=device)
    x_val2 = torch.FloatTensor(x_val2, device=device)
    y_val2 = torch.LongTensor(y_val2, device=device)

# Define layers in model
model = torch.nn.Sequential(
          torch.nn.Linear(D_in, H),
          torch.nn.ReLU(),
          torch.nn.BatchNorm1d(H),
          torch.nn.Linear(H, H),
          torch.nn.ReLU(),
          torch.nn.BatchNorm1d(H),
          torch.nn.Linear(H, H),
          torch.nn.ReLU(),
          torch.nn.BatchNorm1d(H),
          torch.nn.Linear(H, H),
          torch.nn.ReLU(),
          torch.nn.BatchNorm1d(H),
          torch.nn.Linear(H, 2),
          torch.nn.LogSoftmax()
        ).to(device)

# Load Saved Model
if LOAD_SAVED_MODEL_FLAG == 1:
    model.load_state_dict(torch.load(MODEL_FILE_IN))

# Define Loss Function
loss_fn = torch.nn.CrossEntropyLoss()

def accuracy2(out, labels):
  outputs = np.argmax(out, axis=1)
  return np.divide(np.sum(outputs==labels),labels.size)

#Setup optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

if TRAIN_FLAG == 1:
    for t in range(NUMEPOCHS):

        #Forward pass, predict label y given training data x
        y_pred = model(x)

        #Switch to Eval Mode for Validation Data and predict based on validation data x_val
        model.eval()
        y_pred_val = model(x_val)
        y_pred_val2 = model(x_val2)

        #Switch Back to Train Mode
        model.train()

        # Compute the loss and accuracy
        loss = loss_fn(y_pred, y)
        accuracy = accuracy2(y_pred.detach().cpu().numpy(), y.detach().cpu().numpy())
        acc_val = accuracy2(y_pred_val.detach().cpu().numpy(), y_val.detach().cpu().numpy())
        acc_val2 = accuracy2(y_pred_val2.detach().cpu().numpy(), y_val2.detach().cpu().numpy())
        print(t, 'Train Loss=', loss.item(), 'Train Acc:', accuracy, 'Val Acc: ', acc_val, 'Val Acc2: ', acc_val2)
        
        # Zero the gradients before running the backward pass.
        model.zero_grad()
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to all the learnable parameters
        loss.backward()

        # Update Parameters
        optimizer.step()

        # Update the weights using gradient descent
        with torch.no_grad():
            for param in model.parameters():
                param.data -= learning_rate * param.grad

        # Save model
        torch.save(model.state_dict(), MODEL_FILE_OUT)

#Produce Predictions of Test Data Set Pairings
obs, tags = loadDataTags(test_data, test_tags)

#Produce Test Doubles
x_test = []
z = []
for i in range(len(obs)):
    for j in range(len(obs)):
        if i != j:
            d = [obs[i][0:11], obs[j][0:11]]
            datapoint = augmentAndVectorize(d)
            x_test.append(datapoint)
            z.append((i,j))

print('Test Data Points', len(obs), len(x_test))

#Make Test Predictions
x_test = torch.FloatTensor(x_test, device=device)
y_pred = model(x_test)

#Match Predictions
d = defaultdict(float) #Default Dictionary
matches = []
for q in range(len(y_pred)):
    p = np.exp(y_pred[q].detach().cpu().numpy())  
    d[z[q]] = p[0] #Note inverse doublet has same answer from NN
    if p[0] > 0.90:
        matches.append(z[q])
        print('match', d[z[q]], z[q])

#Search for Likely Triplets
tri = []
i = 0
while i < len(z):
    a = z[i][0]
    j = i
    high = []
    while a == z[j][0]:
        b = z[j][1]
        if d[(a,b)] > 0.9:
            high.append((a,b))
        #print('ab', a, b)
        j += 1
        if j == len(z):
            break
    i = j

    #Check Triplet Combinations   
    for q in range(len(high)):
        for r in range(len(high)):
            if high[q] != high[r]:
                sm = d[high[q]] + d[high[r]] + d[high[q][1],high[r][1]]
                #print('high', d[high[q]], d[high[r]], d[high[q][1],high[r][1]])
                if sm>(0.96*3):
                    print('high', sm, high[q][0], high[q][1], high[r][1])
                    tri.append([high[q][0], high[q][1], high[r][1]])

#Eval Versus Truth
print('len tri', len(tri))
correct = 0
falsePos = 0
satsFound = []
for h in tri:
    print('h', h, tags[h[0]], tags[h[1]], tags[h[2]])
    if tags[h[0]] == tags[h[1]] and tags[h[0]] == tags[h[2]]:
        correct += 1
        if tags[h[0]] not in satsFound:
            satsFound.append(tags[h[0]])
    else:
        falsePos += 1

#Get Number of Satellites w/ Any Valid Triplet
sats = []
for t in tags:
    if t not in sats and tags.count(t)>=3:
        sats.append(t)

print('')
print('Correct Triplets: ', correct)
print('False Positive Triplets: ', falsePos)
print('Correct Percent', correct/(correct+falsePos)*100)
print('')
print('Unique Satellites Identified: ', len(satsFound))
print('Total RSOs w/ Triplets: ', len(sats))
print('Percent Found: ', len(satsFound)/len(sats)*100)
print('')
print('Aggregate Score (Max=1.0): ', (correct/(correct+falsePos) + len(satsFound)/len(sats))/2)