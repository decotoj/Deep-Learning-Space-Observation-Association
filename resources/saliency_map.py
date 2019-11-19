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

# matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Config
MODEL_PATH = 'model.pt' #save path for model file
test_data = 'test_data.csv'
test_tags = 'test_tags.csv'

# Define Neural Network Layers (matching pretrained model)
model = trnMod.defineModel()
device = torch.device('cpu') #Assign to CPU

# Load Pre-Trained Neural Network Model
model.load_state_dict(torch.load(MODEL_PATH))
model.eval() # Switch Model to Eval Mode

# Assign model to device (CPU if local)
model = model.to(device)

# Parse Test Data
obs, tags = trnMod.loadDataTags(test_data, test_tags)
print('# Test Set Obs: ', len(obs))
print('# Unique RSOs: ', len(set(tags)))

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
    
    saliency = None
    ##############################################################################
    # TODO: Implement this function. Perform a forward and backward pass through #
    # the model to compute the gradient of the correct class score with respect  #
    # to each input image. You first want to compute the loss over the correct   #
    # scores (we'll combine losses across a batch by summing), and then compute  #
    # the gradients with a backward pass.                                        #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = model(X)  #forward pass    
    score = out.gather(1, y.view(-1, 1)).squeeze() #score for truth class
    score.backward(torch.ones(score.shape)) #backward pass
    grad = X.grad #get gradients
    grad = grad.abs() #absolute value of gradients
    saliency,_ = torch.max(grad, dim=1) #max across input channels

    #TEST1234
    saliency = grad # TESTE1234
    #print('test A', grad)
    #TEST1234

    #NOTE: Explanation of why argument is needed to be passed to 'torch.backward()'
    #https://discuss.pytorch.org/t/loss-backward-raises-error-grad-can-be-implicitly-created-only-for-scalar-outputs/12152
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return saliency

def show_saliency_maps(X, y):

    print(type(X))

    # Convert X and y from numpy arrays to Torch Tensors
    #X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0)
    X_tensor = torch.FloatTensor(X, device=device) #X[0] #torch.from_numpy(X[0]).float().to(device)
    y_tensor = torch.LongTensor(y)

    # Compute saliency maps for images in X
    saliency = compute_saliency_maps(X_tensor, y_tensor, model)

    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency maps together.
    saliency = saliency.numpy()
    N = X.shape[0]
   
    for i in range(N):
        plt.subplot(2, N, i + 1)
        
        #rotated_img = ndimage.rotate(X[i].detach().numpy().swapaxes(0,2), 90)
        #rotated_img = ndimage.rotate(X_tensor[i].detach().numpy().swapaxes(0,2), 90)
        
        x = X[i]
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
        x = asarray(x)
        print('C', x.shape, type(x))

        rotated_img = Image.fromarray((x).astype(np.uint8))

        plt.imshow(rotated_img)
        plt.axis('off')
        plt.subplot(2, N, N + i + 1)

        x = saliency[i]
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
        x = asarray(x)
        print('C sal', x.shape, type(x))
        saliency_img = Image.fromarray((x).astype(np.uint8))

        plt.imshow(saliency_img, cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(12, 8)

        #print('shapes', rotated_img.shape, saliency[i].shape)

    plt.show()


i = 300
j = 202
datapoint = trnMod.augmentAndVectorize([obs[i][:], obs[j][:]]) 

inputs = np.array([datapoint])
labels = np.array([1])

print('inputs', inputs.shape)
show_saliency_maps(inputs, labels)