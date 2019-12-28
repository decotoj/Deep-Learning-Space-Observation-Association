import math
import random
from datetime import timedelta

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from main_synthesize_data import (
    EARTH_ORIENTATION,
    EARTH_ORIENTATION_EPOCH,
    EARTH_ROTATION,
    generate_observers,
    intertial_grid,
    randomEarthObserver,
    rot_z,
    start_epoch,
)

N_OBSERVERS = 250
POINT_INTERVAL = 5  # seconds

N_SECONDS = 86400 // 2
GEO_STATE = (
    [22781.306424624, 35344.771318995, 3092.266805328],
    [-2.587242466, 1.654927728, 0.144787415],
)

N_SECONDS = 86000 // 8
LEO_STATE = (
    [4282.90009, -4989.98448, 1705.55627],
    [4.692897739, 2.062712535, -5.696416645],
)

SC_STATE = GEO_STATE

# constants
EARTH_MU = 398600.4418


def mag(v):
    return math.sqrt(sum([x * x for x in v]))


def norm(v):
    return scale(1 / mag(v), v)


def scale(s, v):
    return [s * x for x in v]


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def gravity(v):
    m = mag(v)
    a = -EARTH_MU / (m * m)
    n = norm(v)
    return scale(a, n)


def integrate(state, dt):
    r0, v0 = state
    a0 = gravity(r0)
    r1 = add(add(r0, scale(dt, v0)), scale(0.5 * dt ** 2, a0))
    a1 = gravity(r1)
    v1 = add(v0, scale(0.5 * dt, add(a0, a1)))
    return (r1, v1)


def hsv_to_rgb(h, s, v):
    c = v * s
    hp = h / 60
    x = c * (1 - abs((hp % 2) - 1))
    if 0 <= hp <= 1:
        return (c, x, 0)
    if 1 < hp <= 2:
        return (x, c, 0)
    if 2 < hp <= 3:
        return (0, c, x)
    if 3 < hp <= 4:
        return (0, x, c)
    if 4 < hp <= 5:
        return (x, 0, c)
    if 5 < hp <= 6:
        return (c, 0, x)
    return (0, 0, 0)


def colormap(n):
    return [hsv_to_rgb(i * (240 / n), 1, 1) for i in range(n)]


if __name__ == "__main__":
    generate_observers(N_OBSERVERS)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    # fig = plt.figure(figsize=(12, 6))
    # ax = fig.add_subplot(111)
    points = []
    sc = SC_STATE
    for t in range(N_SECONDS):
        sc = integrate(sc, 1)
        if t % POINT_INTERVAL == 0:
            observer = randomEarthObserver(sc[0], t)
            if observer is None:
                continue
            t_delta = (
                (start_epoch + timedelta(seconds=t)) - EARTH_ORIENTATION_EPOCH
            ).total_seconds()
            theta = EARTH_ORIENTATION + (t_delta * EARTH_ROTATION)
            observer_fixed = rot_z(observer, theta)
            points.append(observer_fixed)
            # phi, lam = intertial_grid(observer_fixed)
            # points.append((math.degrees(lam), math.degrees(phi)))
            # sc_phi, sc_lam = intertial_grid(rot_z(sc[0], theta))
            # print(math.degrees(sc_phi), math.degrees(sc_lam))
    ax.scatter(
        xs=[p[0] for p in points],
        ys=[p[1] for p in points],
        zs=[p[2] for p in points],
        c=colormap(len(points)),
        marker=".",
    )
    ax.set_xlim([-7000, 7000])
    ax.set_ylim([-7000, 7000])
    ax.set_zlim([-7000, 7000])
    # ax.scatter(
    #     x=[p[0] for p in points],
    #     y=[p[1] for p in points],
    #     c=colormap(len(points)),
    #     marker=".",
    # )
    # ax.set_xlim([-180, 180])
    # ax.set_ylim([-90, 90])
    plt.show()
