import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings(action='once')

large = 22
med = 16
small = 12

params = {'axes.titlesize': large,

          'legend.fontsize': med,

          'figure.figsize': (16, 10),

          'axes.labelsize': med,

          'axes.titlesize': med,

          'xtick.labelsize': med,

          'ytick.labelsize': med,

          'figure.titlesize': large}

plt.rcParams.update(params)

plt.style.use('seaborn-whitegrid')

sns.set_style("white")


# %matplotlib inline

def h_plot():
    vals = [["Precision@1", "Recall@10", "MRR"], ["Precision@1", "Recall@10", "MRR"], ["Precision@1", "Recall@10", "MRR"],
            ["Precision@1", "Recall@10", "MRR"], ["Precision@1", "Recall@10", "MRR"]]
    # Environment: albert           roberta             bert                electra
    # Weight = [[0.408, 0.550, 0.716], [0.462, 0.600, 0.744], [0.490, 0.620, 0.759], [0.492, 0.635, 0.777]]
    # Science
    # Weight = [[0.513, 0.612, 0.825], [0.494, 0.621, 0.839], [0.544, 0.646, 0.846], [0.578, 0.675, 0.853]]
    # Food
    # Weight = [[0.399, 0.661, 0.786], [0.42, 0.678, 0.807], [0.423, 0.663, 0.786], [0.523, 0.69, 0.807], [0.553, 0.724, 0.818]]
    # Weight = [[0.234, 0.526, 0.709], [0.325, 0.528, 0.683], [0.506, 0.611, 0.9], [0.545, 0.618, 0.93],
    #          [0.571, 0.641, 0.918]]
    Weight = [[0.408, 0.492, 0.761], [0.460, 0.534, 0.768], [0.605, 0.706, 0.862], [0.618, 0.688, 0.865],
             [0.632, 0.692, 0.840]]
    mpl.rc('xtick', labelsize=15)

    plt.figure(figsize=(6, 4))

    colors = [plt.cm.Spectral(float(0.25*i)) for i in range(len(vals))]
    # colors = [plt.cm.Spectral(i / float(len(vals) - 1)) for i in range(len(vals))]
    colors[2] = plt.cm.Spectral(0.40)

    n, bins, patches = plt.hist(vals, [0, 1, 2, 3], stacked=False, density=False, weights=Weight,
                                color=colors[:len(vals)], rwidth=0.7)

    # Decoration

    plt.legend({group: col for group, col in zip(["ALBERT", "ELECTRA", "DistilBERT", "RoBERTa", "BERT"], colors[:len(vals)])},
               fontsize=12)

    # plt.title(f"Stacked Histogram of ${x_var}$ colored by ${groupby_var}$", fontsize=22)

    # plt.xlabel(x_var)

    # plt.ylabel("Frequency")

    # plt.ylim(0, 40)
    # lbs = np.unique(df[x_var]).tolist()
    # mpl.rc('ytick', labelsize=25)
    align = " " * 27
    plt.xticks(ticks=bins, labels=[align + "Acc", align + "MRR", align + "Wu&P", ""], rotation=0, ha="center")
    plt.ylim((0.1, 1))
    # plt.show()
    plt.tight_layout()
    plt.savefig("LM_science_new2.pdf")


def w_plot():
    plt.figure(figsize=(6, 5))
    x = [i for i in range(10)]
    #     [0.01, 0.1  , 0.2  , 0.3  , 0.4  , 0.5 , 0.6, 0.8 , 1    , 5]
    Precision = [0.046, 0.085, 0.136, 0.204, 0.250, 0.295, 0.328, 0.348, 0.364, 0.374]
    Recall = [0.798, 0.701, 0.710, 0.774, 0.780, 0.808, 0.828, 0.835, 0.847, 0.848]
    F1 = [0.087, 0.151, 0.229, 0.323, 0.379, 0.432, 0.470, 0.492, 0.509, 0.519]
    # wu_p = [0.842, 0.843, 0.846, 0.852, 0.847, 0.845, 0.849, 0.847, 0.858, 0.834]
    # x = np.arange(20, 350)
    colors = [plt.cm.Spectral(i / float(5)) for i in range(5)]
    l1 = plt.plot(x, Precision, 'ro-', label='Precision', color=colors[0])
    l2 = plt.plot(x, Recall, 'g+-', label='Recall', color=colors[1])
    l3 = plt.plot(x, F1, 'b^-', label='F1', color=colors[4])
    # plt.plot(x, wu_p, 'ro-', x, mrr, 'g+-', x, acc, 'b^-')
    plt.xticks(ticks=x, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # plt.title('The Lasers in Three Conditions')
    plt.xlabel('epoch')
    # plt.ylabel('F1')
    #plt.legend(loc='center', ncol=3, bbox_to_anchor=(0.5, 0.65))
    plt.legend()
    # plt.ylim((0.45, 0.9))
    # plt.show()
    plt.savefig("detection_result.pdf")


def s_plot():
    plt.figure(figsize=(6, 5))
    x = [i for i in range(7)]
    #      [0,     1  ,   2  ,   3  ,   4  ,   5 ,   >5]
    acc =  [0.158, 0.440, 0.480, 0.490, 0.458, 0.491, 0.553]
    mrr =  [0.299, 0.565, 0.612, 0.615, 0.601, 0.634, 0.688]
    wu_p = [0.709, 0.825, 0.842, 0.834, 0.809, 0.812, 0.842]
    # x = np.arange(20, 350)
    colors = [plt.cm.Spectral(i / float(5)) for i in range(5)]
    l1 = plt.plot(x, wu_p, 'ro-', label='wu&p', color=colors[0])
    l2 = plt.plot(x, mrr, 'g+-', label='mrr', color=colors[1])
    l3 = plt.plot(x, acc, 'b^-', label='acc', color=colors[4])
    # plt.plot(x, wu_p, 'ro-', x, mrr, 'g+-', x, acc, 'b^-')
    plt.xticks(ticks=x, labels=["0", "1", "2", "3", "4", "5", ">5"])
    # plt.title('The Lasers in Three Conditions')
    plt.xlabel('Number of sibling nodes')
    plt.ylabel('performance')
    plt.legend(loc='best')
    # plt.ylim((0.45, 0.9))
    # plt.show()
    plt.savefig("sibling.pdf")


if __name__ == '__main__':
    w_plot()
