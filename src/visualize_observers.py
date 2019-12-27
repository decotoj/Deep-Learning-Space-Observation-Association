import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from main_synthesize_data import randomEarthObserver

N_ITER = 750
AX_LIMIT = [-7000, 7000]
SC_LOCATION = [6378 + 35786, 0, 0]

if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    points = []
    for _ in range(N_ITER):
        points.append(randomEarthObserver(SC_LOCATION))
    ax.scatter(
        xs=[p[0] for p in points],
        ys=[p[1] for p in points],
        zs=[p[2] for p in points],
        marker="."
    )

    ax.set_xlim(AX_LIMIT)
    ax.set_ylim(AX_LIMIT)
    ax.set_zlim(AX_LIMIT)
    plt.show()
