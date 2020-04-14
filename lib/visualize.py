import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')

def draw_distribution_comparison(dis_1, dis_2, figname, 
        logger, prefix='', output_path='.',
        figsize=(36,24),
        title_size=36, tick_size=36,
        legend_size=44, label_size=36, alpha=0.5,
        same_color='b', not_same_color='r',
        bin_size=40, label_1='label_1', label_2='label_2',
        title='', xlabel=''):
    """
    Draw the distribution of comparison and distance

    :param distances: pairwise distance matrix
    :param identical: pairwise comparison
    :param comparison_type: subject or syndrome
    :param metric: euclidean or cosine distance
    :param logger: logger
    :param prefix: prefix for title and filename
    :param output_path: output folder
    :param title_size: fontsize of title
    :param tick_size: fontsize of tick
    :param legend_size: fontsize of legend
    :param label_size: fontsize of label
    """
    # get the array of match/not-match distances
    logger.info("Plotting distribution histogram")
    max_dist = max(max(dis_1), max(dis_2))

    bins = np.linspace(0, max_dist, bin_size)
    fig = plt.figure(figsize=(16, 12))
    n, bins, patches = plt.hist(dis_1, bins, facecolor=same_color,
            alpha=alpha, label=label_1)
    n, bins, patches = plt.hist(dis_2, bins,
            facecolor=not_same_color, alpha=alpha,
            label=label_2)
    plt.legend(prop={'size': legend_size})
    plt.title('{}'.format(title), fontsize=title_size)
    plt.xlabel('{}'.format(xlabel), fontsize=label_size)
    plt.ylabel('Count', fontsize=label_size)
    #plt.yscale('log')
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.grid(True)
    figname = os.path.join(output_path, figname)
    logger.info("Figure save to {}".format(figname))
    plt.savefig(figname)
