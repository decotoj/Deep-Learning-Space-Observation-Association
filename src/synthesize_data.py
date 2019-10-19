import twobody as twobody
import random
import numpy as np
import math

#Produce a set of N Optical Observations of RSOs by Randomely Placed Earth Based Observers

#Settings
N = 100000 #Number of simulated data points in each set
nLow = 3 #Low end of range for # of obs of a given RSO
nHigh = 10 #High end of range for # of obs of a given RSO
dataTags = ['train', 'val', 'test']
tf = 3600*12 #Time range of observation is between t=0 and t=tf (seconds)
StreakLength = 120 #Length of time for each observation from start of collect to end (seconds)
Pstd = 0.1 #Standard Deviation of Gaussian Error on RSO Position (km) - Causes Observation Angles to Be Off Truth

#Constants
mu = 398600.4418 #Earth Gravitional Constant km^2/s^2

def angleBetweenVectors(v1, v2):
  return math.acos(np.dot(v1,v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def randomEarthObserver(p):

    while True:

        #Build an Observer Positioned Randomely on Earth (+/- 500 meters from Sea Level)
        R = random.uniform(6378-0.5, 6378+5)
        P = [random.random()-0.5 for q in range(3)]
        m = np.linalg.norm(P)
        P = [q/m*R for q in P]

        #Ensure Randomly Placed Observer Isn't Looking Through Earth to Observe RSO
        C = [p[i]-P[i] for i in range(3)] #Vector from Observer to RSO
        alpha = angleBetweenVectors(C, P) #Angle Off Horizon of Observation Vector (Must be <1.57 radians)
        if alpha < 1.57:
            return P

def RandomObservationsN(NumVec, s):

    #Create Random Observations
    t = []
    ob = []
    for i in range(0,NumVec): 
         
         t = random.uniform(0,tf) #Random time between 0 and tf
         p, v = s.posvelatt(t)
         p2, v2 = s.posvelatt(t+StreakLength)

         P = randomEarthObserver(p) #Randomly Positioned Earth Observer

         #Add Random Error to RSO Position Knowledge
         p = [q+random.gauss(0,Pstd) for q in p]
         p2 = [q+random.gauss(0,Pstd) for q in p2]

         u = [p[0] - P[0], p[1] - P[1], p[2] - P[2]] #RSO Position at Random Index Relative to Observer
         m = np.linalg.norm(u) 
         u = [q/m for q in u] #Observation Unit Vector

         u2 = [p2[0], p2[1], p2[2]]
         m = np.linalg.norm(u2)
         u2 = [q/m for q in u2]

         StreakDirection = [u2[0]-u[0], u2[1]-u[1],u2[2]-u[2]]

         ob.append([float(t)] + P + u + StreakDirection) #10 params
        
    return ob

def randomSatellite():

    s = twobody.TwoBodyOrbit("RSO", mu=mu)   # create an instance

    a = random.uniform(41164,43164) #semi-major axis
    e = random.uniform(0,0.1)  #eccentricity
    i = random.uniform(0,20) #inclination
    LoAN = random.uniform(0,360) #longitude of ascending node
    AoP = random.uniform(0,360) #argument of perigee
    MA = random.uniform(0,360) #mean anomaly
    
    s.setOrbKepl(0, a, e, i, LoAN, AoP, MA=MA) # define the orbit

    return s

def synthesizeDataset(N, nLow, nHigh, outFile1, outFile2):

    #Create Data
    obs = []
    tags = []
    label = 0
    while len(obs) < N:
        n = random.randint(nLow, nHigh)
        obs = obs + RandomObservationsN(n, randomSatellite()) #Grab n Random Observations
        tags = tags + [label]*n
        label += 1
    obs = obs[0:N] #Cut to Desired Length
    tags = tags[0:N] #Cut to Desired Length
    print('obs: ', len(obs), len(tags))

    #Randomize the order of the observations
    c = list(zip(obs, tags))
    random.shuffle(c)
    obs, tags = zip(*c)

    #Write the observation and tag data to files
    with open(outFile1, 'w') as f:
        for q in obs:
            q = [str(r) for r in q]
            f.write(','.join(q) + '\n')
    with open(outFile2, 'w') as f:
        [f.write(str(q) + '\n') for q in tags]

#Synthesize Train, Validation and Test Data
[synthesizeDataset(N, nLow, nHigh, q + '_data.csv', q + '_tags.csv') for q in dataTags]