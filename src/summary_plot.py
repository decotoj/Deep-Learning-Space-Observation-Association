# import torch
# import random
# import main_train_model as trnMod
# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.offsetbox import AnchoredText
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from scipy import stats

# bins = [1,2,3,4]
# c = [1]*74
# c = c + [2]*340
# c = c + [3] * 1340

# # Helper Function for Plotting
# def bins_labels(bins, **kwargs):
#     bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
#     plt.xticks(np.arange(min(bins)+bin_w/2, max(bins), bin_w), bins, **kwargs)
#     plt.xlim(bins[0], bins[-1])

# # Plot
# fig, ax = plt.subplots()
# ax.grid()
# ax.set(xlabel='Number of Triplets Explored', ylabel='Number of RSOs',title='Number of Unique RSOs Correctly Identified in At Least One Triplet')
# sns.distplot(c, bins=bins, kde=False, hist_kws=dict(edgecolor="k", linewidth=2))
# bins_labels(bins, fontsize=20)
# fig.savefig("summary_plot.png")
# plt.draw()
# plt.show()


import matplotlib.pyplot as plt
import numpy as np

category_names = ["Acquired", "Missed"]
results = {
    "N=1e3\n0.0006% Searched": [27, 142 - 27],
    "N=1e4\n0.006% Searched": [74, 142 - 74],
    "N=1e5\n0.06% Searched": [111, 142 - 111],
    "N=1.66e8\n100% Searched": [142, 0],
}


def survey(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap("RdYlGn")(
        np.linspace(0.85, 0.15, data.shape[1])
    )

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())
    fig.suptitle(
        "Number of Acquired RSOs in Test Set \n w/ Varying Number of Explored Triplet Search Solutions",
        fontsize=12,
    )
    ax.set_ylabel(
        "Number of Exlored Triplets from Search Space", fontsize="medium"
    )  # relative to plt.rcParams['font.size']

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(
            labels, widths, left=starts, height=0.5, label=colname, color=color
        )
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = "white" if r * g * b < 0.5 else "darkgrey"
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            ax.text(
                x, y, str(int(c)), ha="center", va="center", color=text_color
            )
    ax.legend(
        ncol=len(category_names),
        bbox_to_anchor=(0, 1),
        loc="lower left",
        fontsize="small",
    )

    return fig, ax


survey(results, category_names)
plt.show()
