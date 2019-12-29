import math
import random
from datetime import datetime, timedelta

import numpy as np

import twobody as twobody

# Produce a set of N Optical Observations of RSOs by Randomely Placed Earth Based Observers

# Settings
N = 100000  # 100000 #Number of simulated data points in each set (Baseline = 100000)
nLow = 3  # Low end of range for # of obs of a given RSO
nHigh = 10  # High end of range for # of obs of a given RSO
dataTags = ["train", "val", "test"]  # ['train', 'val', 'test']
tf = 3600 * 12  # Time range of observation is between 0 to tf (seconds)
StreakLength = 120  # Length of time for each observation from start of collect to end (seconds)
Pstd = 0.1  # Standard Deviation of Gaussian Error on RSO Position (km) - Introduces Error on Observation Angles
n_observers = 250
start_epoch = datetime(
    2019, 12, 27, 0, 0, 0
)  # model ground station position starting on this date/time

# Constants
mu = 398600.4418  # Earth Gravitional Constant km^2/s^2
EARTH_RADIUS = 6378.137  # kilometers
EARTH_FLAT = 1.0 / 298.257  # unitless
EARTH_ECC_SQ = EARTH_FLAT * (2 - EARTH_FLAT)  # unitless
EARTH_ROTATION = 7.292115855300805e-5  # radians per second
EARTH_ORIENTATION = 1.7510838783001768  # radians
EARTH_ORIENTATION_EPOCH = datetime(2015, 1, 1)
OBSERVERS = []  # ecef km positions


def angleBetweenVectors(v1, v2):
    return math.acos(
        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    )


def intertial_grid(v):
    sma = EARTH_RADIUS
    esq = EARTH_ECC_SQ
    x, y, z = v
    lon = math.atan2(y, x)
    r = math.sqrt(x * x + y * y)
    phi = math.atan(z / r)
    lat = phi
    c = 0.0
    for _ in range(6):
        slat = math.sin(lat)
        c = 1.0 / math.sqrt(1 - esq * slat * slat)
        lat = math.atan((z + sma * c * esq * slat) / r)
    return lat, lon


def hypot(v):
    return math.sqrt(sum([x * x for x in v]))


def half_angle(v):
    return math.acos(EARTH_RADIUS / hypot(v))


def rot_z(v, theta):
    x, y, z = v
    cosT = math.cos(theta)
    sinT = math.sin(theta)
    return [
        cosT * x + sinT * y + 0 * z,
        -sinT * x + cosT * y + 0 * z,
        0 * x + 0 * y + 1 * z,
    ]


def clamp(n, min_val, max_val):
    return max(min_val, min(n, max_val))


def haversine(phi_1, lam_1, phi_2, lam_2):
    a = math.sin((phi_2 - phi_1) / 2) ** 2
    b = math.cos(phi_1) * math.cos(phi_2) * math.sin((lam_2 - lam_1) / 2) ** 2
    return 2.0 * math.asin(math.sqrt(clamp(a + b, -1.0, 1.0)))


def generate_observers(n):
    global OBSERVERS
    OBSERVERS = []
    for _ in range(n):
        R = random.uniform(6378 - 0.5, 6378 + 0.5)
        P = [random.gauss(0, 1) for _ in range(3)]
        m = np.linalg.norm(P)
        OBSERVERS.append([(q / m) * R for q in P])


def randomEarthObserver(p, t):
    obserers_shuffled = OBSERVERS[:]
    random.shuffle(obserers_shuffled)
    for ecef in obserers_shuffled:
        t_delta = (
            (start_epoch + timedelta(seconds=t)) - EARTH_ORIENTATION_EPOCH
        ).total_seconds()
        theta = EARTH_ORIENTATION + (t_delta * EARTH_ROTATION)
        P = rot_z(ecef, -theta)
        ha = half_angle(p)
        ig_p = intertial_grid(p)
        ig_P = intertial_grid(P)
        if haversine(*ig_p, *ig_P) < ha:
            return P


def RandomObservationsN(NumVec, s):

    # Create Random Observations
    t = []
    ob = []
    for _ in range(NumVec):

        t = random.uniform(0, tf)  # Random time between 0 and tf
        p, _ = s.posvelatt(t)
        p2, _ = s.posvelatt(t + StreakLength)

        P = randomEarthObserver(p, t)  # Randomly Selected Earth Observer

        # Add Random Error to RSO Position Knowledge
        p = [q + random.gauss(0, Pstd) for q in p]
        p2 = [q + random.gauss(0, Pstd) for q in p2]

        u = [
            p[0] - P[0],
            p[1] - P[1],
            p[2] - P[2],
        ]  # RSO Position at Random Index Relative to Observer
        m = np.linalg.norm(u)
        u = [q / m for q in u]  # Observation Unit Vector

        u2 = [
            p2[0] - P[0],
            p2[1] - P[1],
            p2[2] - P[2],
        ]  # WARNING: Ignores observer movement over streak time interval
        m = np.linalg.norm(u2)
        u2 = [q / m for q in u2]

        StreakDirection = [u2[0] - u[0], u2[1] - u[1], u2[2] - u[2]]
        StreakDirection = [q / StreakLength for q in StreakDirection]

        ob.append([float(t)] + P + u + StreakDirection)  # 10 params

    return ob


def randomSatellite():

    s = twobody.TwoBodyOrbit("RSO", mu=mu)  # create an instance

    a = random.uniform(41164, 43164)  # semi-major axis
    e = random.uniform(0, 0.1)  # eccentricity
    i = random.uniform(0, 20)  # inclination
    LoAN = random.uniform(0, 360)  # longitude of ascending node
    AoP = random.uniform(0, 360)  # argument of perigee
    MA = random.uniform(0, 360)  # mean anomaly

    s.setOrbKepl(0, a, e, i, LoAN, AoP, MA=MA)  # define the orbit

    return s


def synthesizeDataset(N, nLow, nHigh, outFile1, outFile2):

    # Create Data
    obs = []
    tags = []
    label = 0
    while len(obs) < N:
        n = random.randint(nLow, nHigh)
        obs = obs + RandomObservationsN(
            n, randomSatellite()
        )  # Grab n Random Observations
        tags = tags + [label] * n
        label += 1
    obs = obs[0:N]  # Cut to Desired Length
    tags = tags[0:N]  # Cut to Desired Length
    print("obs: ", len(obs), len(tags))

    # Randomize the order of the observations
    c = list(zip(obs, tags))
    random.shuffle(c)
    obs, tags = zip(*c)

    # Write the observation and tag data to files
    with open(outFile1, "w") as f:
        for q in obs:
            q = [str(r) for r in q]
            f.write(",".join(q) + "\n")
    with open(outFile2, "w") as f:
        [f.write(str(q) + "\n") for q in tags]


if __name__ == "__main__":
    generate_observers(n_observers)
    # Synthesize Train, Validation and Test Data
    [
        synthesizeDataset(N, nLow, nHigh, q + "_data.csv", q + "_tags.csv")
        for q in dataTags
    ]
