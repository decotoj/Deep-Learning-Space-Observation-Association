# Experiment Training a Neural Network to Aid Optical Space Observation Association
# Last Revision: 10/19/2019
# Authors: Jake Decoto

# Next Steps
#1) Audit code (main and sythesize) and hone down to minimum
#A) Reduce false positive triplets, 1) loss function for doubles penalize false+, 2) separate triplet NN for those identified by doublet NN
#B) Separate out grading and knowledge of tags into separate callable function (so can be run as unknowing participant)
#C) Revisit simulated data settings to ensure data is sufficiently real world problem representive

# The following loads prebuilt training and test data consisting of optical observerations
# of satellites.  A neural network is then trained to identify pairs of optical observations
# in the labelled training data as being of the same satellite or not.  This network is then
# used to identify triplets of observations in the test data that are most likely to be of 
# the same satellite.  No apriori knowledge of physics or orbital mechanics is used.

import torch
import random
import numpy as np
import torch.optim as optim
from collections import defaultdict

random.seed(654654)

#Network and Data Parameters
LOAD_SAVED_MODEL_FLAG = 1 #1=Load prior saved model, 0=Train model from scratch
TRAIN_FLAG = 2 #1=TRAIN, 0=EVAL ONLY, 2=TRAIN BUT DONT SAVE
MODEL_FILE = 'model.pt' #Path to saved model file if applicable, also is name of new model file to be saved
D_in = 416 #Length of Input Vectors after Augmentation (time, observer x-y-z unit vec, observervation x-y-z unit vec, streak direction x-y-z unit vec, observer vector mag)
H = 208 #416 #Hidden layer(s) dimension
N = 1000#Number of training examples to create from loaded data (Baseline = 40000)
Nval = 1000 ##Number of validation examples to create from loaded data (Baseline = 2500)
NUMEPOCHS = 1000 #Number of Training Epochs
learning_rate = 1e-6 #Baseline = 1e-4
train_data = 'train_data.csv'
train_tags = 'train_tags.csv'
val_data = 'val_data.csv'
val_tags = 'val_tags.csv'

def augmentAndVectorize(ob):

    #Sort Observations by Epoch and Normalize Epoch
    ob.sort(key=lambda x: x[0])
    for q in range(len(ob)):
        ob[q][0] = ob[q][0] - ob[0][0]

    # # 600 Parameters w/ ~60% Validation Accuracy (w/ 67% Train Acc) Reached by Epoch 73
    # # Add Derived Parameter from Observation Unit Vector and Streak Vector and Magnitude (416)
    # for z in range(1, len(ob)):
    #     for q in [4,5,6,7,8,9,10,11]:
    #         for r in [4,5,6,7,8,9,10,11]:
    #             for s in [0,4,5,6,7,8,9,10,11]:
    #                 ob[z].append((ob[z][q]-ob[z-1][r])/(ob[z][s]-ob[z-1][s]))

    #416 Parameters w/ ~60% Validation Accuracy (w/ 77% Train Acc) Reached by Epoch 37
    #Add Derived Parameter from Observation Unit Vector and Streak Vector and Magnitude (416)
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
def buildLabelledDataset(Nc, obs, tags):
    x = []
    y = []
    while len(x) <= Nc:

        #Find Matching Double
        while True:
            n = random.randint(0, len(tags)-1) 
            indices = [i for i, x in enumerate(tags) if x == tags[n]]
            if len(indices) > 1:
                d = [obs[indices[0]][:], obs[indices[1]][:]]
                x.append(augmentAndVectorize(d))
                y.append(0)
                break

        #Find Non-Matching Double
        while True:
            n2 = random.randint(0, len(tags)-1) 
            if tags[n] != tags[n2]:
                d = [obs[n][:], obs[n2][:]] 
                x.append(augmentAndVectorize(d))
                y.append(1)
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

def accuracy(out, labels):
  outputs = np.argmax(out, axis=1)
  return np.divide(np.sum(outputs==labels),labels.size)

def defineModel():

    device = torch.device('cpu') #Assign to CPU

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
            torch.nn.LogSoftmax(1) #11/14/2019: Changed to (1) instead of () due to depracation warning, untested
            ).to(device)

    return model


if __name__ == "__main__":

    device = torch.device('cpu') #Assign to CPU

    if TRAIN_FLAG >= 1:

        #Build Validation Dataset
        obs_val, tags_val = loadDataTags(val_data, val_tags) #Load Validation Data 
        x_val, y_val = buildLabelledDataset(Nval, obs_val, tags_val)
        print('Validation Data & # RSOs Used', Nval, len(x_val), len(y_val))

        #Build Training Dataset
        obs, tags = loadDataTags(train_data, train_tags) #Load Training Data
        x, y = buildLabelledDataset(N, obs, tags)
        print('Training Data & # RSOs Used', N, len(x), len(y))

        print('/n Total RSOs', len(set(tags)))

        #Assign Training and Val Data
        x = torch.FloatTensor(x, device=device)
        y = torch.LongTensor(y, device=device)
        x_val = torch.FloatTensor(x_val, device=device)
        y_val = torch.LongTensor(y_val, device=device)

    #Define Model Layers
    model = defineModel()

    # Load Saved Model
    if LOAD_SAVED_MODEL_FLAG == 1:
        model.load_state_dict(torch.load(MODEL_FILE))

    # Define Loss Function
    loss_fn = torch.nn.CrossEntropyLoss()

    #Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

    if TRAIN_FLAG >= 1:
        acc_val_best = 0
        for t in range(NUMEPOCHS):
            model.eval()
            #Forward pass, predict label y given training data x
            y_pred = model(x)

            #Switch to Eval Mode for Validation Data and predict based on validation data x_val
            #model.eval()
            y_pred_val = model(x_val)

            #Switch Back to Train Mode
            model.train()

            # Compute the loss and accuracy
            loss = loss_fn(y_pred, y)
            acc_train = accuracy(y_pred.detach().cpu().numpy(), y.detach().cpu().numpy())
            model.eval()
            acc_val = accuracy(y_pred_val.detach().cpu().numpy(), y_val.detach().cpu().numpy())
            model.train()
            print(t, 'Train Loss=', loss.item(), 'Train Acc:', acc_train, 'Val Acc: ', acc_val)

            if acc_val > acc_val_best and TRAIN_FLAG != 2:
                model.eval()
                torch.save(model.state_dict(), MODEL_FILE) # Save model
                model.train()
                acc_val_best = acc_val
                print('saved model')

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

        