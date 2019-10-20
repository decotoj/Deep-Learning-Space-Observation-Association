import matplotlib
import matplotlib.pyplot as plt
import mathUtil as mUtil
import math
import numpy as np

dataFile = 'train_data.csv'
tagFile = 'train_tags.csv'

#Parse Train Data File
with open(dataFile, 'r') as f:
    obs = [q.split(',') for q in f.readlines()]
for i in range(len(obs)):
    obs[i] = [float(q) for q in obs[i]]

print('# Obs: ', len(obs))

RA = []
DEC = []
RAs = []
DECs = []
for i in range(100):

    ob = obs[i]

    obVec = [float(o) for o in ob[4:7]]
    ra, dec = mUtil.unitVectorToAzEl(obVec)
    RA.append(math.degrees(ra))
    DEC.append(math.degrees(dec))

    obStreak = [float(o) for o in ob[7:10]]
    ra, dec = mUtil.unitVectorToAzEl(obStreak)
    RAs.append(math.degrees(ra)/10)
    DECs.append(math.degrees(dec)/10)

fig, ax = plt.subplots()

ax.scatter(RA, DEC)

ax.set(xlabel='Right Ascension (deg)', ylabel='Declination (deg)', title='Example Observation Vectors w/ Streak Direction and Magnitude')
ax.set_facecolor('k')
ax.grid()

for i in range(len(RA)):
    plt.arrow(RA[i], DEC[i], RAs[i], DECs[i], shape='full', color='r', length_includes_head=False, 
         zorder=0, head_length=3., head_width=1.5)

plt.show()