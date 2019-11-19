# Used to Visualize Simulated Observation Data
# Data Created w/ 'sythesize_data.py'

import matplotlib
import matplotlib.pyplot as plt
import mathUtil as mUtil
import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import PIL
from PIL import Image
from mpl_toolkits.basemap import Basemap

#Config
X = 4 #Number of Example RSOs to Use
dataFile = 'train_data.csv'
tagFile = 'train_tags.csv'

# Parse Tags File
with open(tagFile, 'r') as f:
    tags = [q.split(',')[0].replace('\n','') for q in f.readlines()]

# Find Indices of Obs from X Examples
c = 0
ind = []
while c < X:
    indices = [i for i, x in enumerate(tags) if x == tags[c]]
    ind.append(indices)
    c += 1

# Parse Train Data File
with open(dataFile, 'r') as f:
    obs = [q.split(',') for q in f.readlines()]
for i in range(len(obs)):
    obs[i] = [float(q) for q in obs[i]]

#######################################################################
# 2D RA/DEC w/ Streak Length Plot

fig, ax = plt.subplots()

# Grab Relevant Obs for X RSOs
for indices in ind:

    RA = []
    DEC = []
    RAs = []
    DECs = []
    for i in indices:

        ob = obs[i]

        obVec = [float(o) for o in ob[4:7]]
        ra, dec = mUtil.unitVectorToAzEl(obVec)
        RA.append(math.degrees(ra))
        DEC.append(math.degrees(dec))

        obStreak = [float(o) for o in ob[7:10]]
        ra, dec = mUtil.unitVectorToAzEl(obStreak)
        RAs.append(math.degrees(ra)/10)
        DECs.append(math.degrees(dec)/10)


    #2D Plot Showing RA/DEC of Observations and Streak Direction and Approximate Magnitude
    ax.scatter(RA, DEC)
    for i in range(len(RA)):
        plt.arrow(RA[i], DEC[i], RAs[i], DECs[i], shape='full', color='r', length_includes_head=False, 
            zorder=0, head_length=3., head_width=1.5)

ax.set(xlabel='Right Ascension (deg)', ylabel='Declination (deg)', title='Example Observation Vectors and Streaks for 4 RSOs')
ax.set_facecolor('k')
ax.grid()

#######################################################################

#3D Plot Showing a Single Observation

# load bluemarble with PIL
bm = PIL.Image.open('resources/bluemarble.jpg')
# it's big, so I'll rescale it, convert to array, and divide by 256 to get RGB values that matplotlib accept 
bm = np.array(bm.resize([int(d/5) for d in bm.size]))/256.

# coordinates of the image - don't know if this is entirely accurate, but probably close
lons = np.linspace(-180, 180, bm.shape[1]) * np.pi/180 
lats = np.linspace(-90, 90, bm.shape[0])[::-1] * np.pi/180 

# repeat code from one of the examples linked to in the question, except for specifying facecolors:
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
fig.set_facecolor('black')
ax.set_facecolor('black') 
ax.grid(False) 
ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

x = np.outer(np.cos(lons), np.cos(lats)).T
y = np.outer(np.sin(lons), np.cos(lats)).T
z = np.outer(np.ones(np.size(lons)), np.sin(lats)).T
ax.plot_surface(x, y, z, rstride=4, cstride=4, facecolors = bm)

# Grab Relevant Obs for X RSOs
for i in range(10):
    ob = obs[i]

    P = [float(o) for o in ob[1:4]]
    m = np.linalg.norm(P) 
    Punit = [q/m for q in P] #Observer Unit Vector

    obVec = [float(o) for o in ob[4:7]]
    obStreak = [float(o) for o in ob[7:10]]

    ax.quiver(Punit[0], Punit[1], Punit[2], obVec[0], obVec[1], obVec[2], length=2, normalize=True)

plt.show()

#######################################################################