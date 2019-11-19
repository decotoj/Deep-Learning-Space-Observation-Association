# Build a Composite Saliency Map 
# 11/17/2019
# Jake Decoto

import torch
import torchvision
import torchvision.transforms as T
import random
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, models, transforms
import os
import torch.nn as nn
from scipy import ndimage
import main_train_model as trnMod
from numpy import asarray
from matplotlib import cm

# Config
MODEL_PATH = 'model.pt' #save path for model file
test_data = 'test_data.csv'
test_tags = 'test_tags.csv'

# Plot Configure
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Define Neural Network Layers (matching pretrained model)
model = trnMod.defineModel()
device = torch.device('cpu') #Assign to CPU

# Load Pre-Trained Neural Network Model
model.load_state_dict(torch.load(MODEL_PATH))
model.eval() # Switch Model to Eval Mode
model = model.to(device) # Assign model to device (CPU if local)

# Parse Test Data
obs, tags = trnMod.loadDataTags(test_data, test_tags)
print('# Test Set Obs: ', len(obs))
print('# Unique RSOs: ', len(set(tags)))

def transformTo2121(x):
    b = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    x = np.append(x, b)
    t = list(x)
    mn = min(t)
    mx = max(t)
    t = [int((q-mn)/(mx-mn)*255) for q in t]
    x = []
    for q in range(21):
        i1 = 0 + q*21
        i2 = 21 + q*21
        x.append(t[i1:i2])
    x = np.array(x)
    return asarray(x)

def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode
    model.eval()
    
    # Make input tensor require gradient
    X.requires_grad_()
    
    saliency = []#None

    ##############################################################################
    # Perform a forward and backward pass through #
    # the model to compute the gradient of the correct class score with respect  #
    # to each input image. You first want to compute the loss over the correct   #
    # scores (we'll combine losses across a batch by summing), and then compute  #
    # the gradients with a backward pass.                                        #
    ##############################################################################

    out = model(X)  #forward pass    
    score = out.gather(1, y.view(-1, 1)).squeeze() #score for truth class
    score.backward(torch.ones(score.shape)) #backward pass
    grad = X.grad #get gradients
    grad = grad.abs() #absolute value of gradients
    #saliency,_ = torch.max(grad, dim=1) #max across input channels
    saliency = grad #do not max over input channels since only 1 channel

    return saliency

def show_saliency_maps(X, y):

    # Convert X and y from numpy arrays to Torch Tensors
    X_tensor = torch.FloatTensor(X, device=device) #X[0] #torch.from_numpy(X[0]).float().to(device)
    y_tensor = torch.LongTensor(y)

    # Compute saliency maps for images in X
    saliency = compute_saliency_maps(X_tensor, y_tensor, model)

    # Convert the saliency map from Torch Tensor to numpy array and show images and saliency maps together.
    saliency = saliency.numpy()
    N = X.shape[0]
   
    d = 3

    # Step Through Results
    composite = []
    for i in range(len(saliency)):

        if i < d:
            print('i', i)
            plt.subplot(2, d, i + 1)
            # Image
            x = transformTo2121(X[i])
            img = Image.fromarray((x).astype(np.uint8))
            plt.title('Image')    
            plt.imshow(img)
            plt.axis('off')
            plt.subplot(2, d, d + i + 1)

            # Saliency Map
            x = transformTo2121(saliency[i])
            saliency_img = Image.fromarray((x).astype(np.uint8))
            plt.title('Saliency Map')
            plt.imshow(saliency_img, cmap=plt.cm.hot)
            plt.axis('off')
            plt.gcf().set_size_inches(12, 8)

        composite.append(x) #Add Current Saliency Map to Composite

    #plt.show() #Show Grid of Individual Saliency Maps

    # Process Composite
    x = composite[0]
    for q in range(1,len(composite)):
        x = np.add(x,composite[q])
    print('test1234', len(composite))

    # for i in range(10):
    #     x[0][i] = 255*3

    i = len(composite)
    x = np.divide(x, i)

    saliency_img = Image.fromarray((x).astype(np.uint8))

    #print('x min max', min(x.all()), max(x.all()))

    # Plot
    fig, ax = plt.subplots()
    ax.set(title='Composite Saliency Map')
    plt.imshow(saliency_img, cmap=plt.cm.hot)
    plt.axis('off')
    plt.gcf().set_size_inches(12, 8)
    plt.show()

pairs = []
labels = []

# Grab Given Number of Examples of Either Match or Non Match
import random
while len(pairs) < 100:
    i = int(random.uniform(0,len(obs)))
    j = int(random.uniform(0,len(obs)))
    if i != j:
            # if tags[i] == tags[j]:
            #     pairs.append((i,j))
            #     labels.append([0])
            if tags[i] != tags[j]:
                pairs.append((i,j))
                labels.append([1])
    if len(pairs) == 100:
        break



inputs = []
for i,j in pairs:
    datapoint = trnMod.augmentAndVectorize([obs[i][:], obs[j][:]]) 
    inputs.append(datapoint)
inputs = np.array(inputs)
labels = np.array(labels)

print('inputs', inputs.shape)
show_saliency_maps(inputs, labels)