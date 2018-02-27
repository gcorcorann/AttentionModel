#!/usr/bin/env python3
"""Read Attention Dataset Annotations.
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def read_labels(labels_path):
    """Read annotated labels and store the counts.

    Args:
        labels_path (string): path to attention annotations
    Returns:
        ndarray: Array holding total count for each label
    """
    data = []
    with open(labels_path, 'r') as f:
        for line in f:
            line = line.split()
            sample = (line[0], int(line[1]))
            data.append(sample)
        
    dtype = [('video', '<U50'), ('label', int)]
    X = np.array(data, dtype=dtype)
    X = np.sort(X, order='video')
    return X

def compute_labels(path_list):
    """Compute the mean and median of given annotations.

    Args:
        path_list (list): list of paths to attention annotations

    Returns:
        list: List of arrays holding final labels.
    """
    labels = []
    videos = []
    for i, path in enumerate(path_list):
        X = read_labels(path)
        for sample in X:
            labels.append(sample[1])
            if i == 0:
                videos.append(sample[0])

    labels = np.array(labels).reshape(len(path_list), -1).T
    
    labels_avg = np.average(labels, 1).round()
    labels_med = np.median(labels, 1)
    return labels_med, labels_avg, videos

def count_labels(labels_path):
    """Compute number of labels in each attention level.

    Args:
        labels_path (string): path to attention labels

    Returns:
        ndarray: array consisting of counts for each label.
    """
    counts = np.zeros(4)
    with open(labels_path, 'r') as f:
        for line in f:
            line = int(line.split()[1]) - 1
            counts[line] += 1

    return counts

def save_labels(labels_array, videos, name):
    videos = np.array(videos)
    annotations = np.vstack((videos, labels_array)).T
    f = open(name, 'w')
    for i in annotations:
        s = str(i[0]) + " " + str(i[1][0]) + "\n"
        f.write(s)

    f.close()


def main():
    """Main Function."""
    # labels datapth
    labels_path = '/home/gary/datasets/accv/labels'

    labels = glob.glob(os.path.join(labels_path, 'labels*.txt'))
    classes = ['Low Attention', 'Medium Attention',
            'High Attention', 'Very High Attention']
    print(labels)
    print(classes)
    print()

    # list to hold counts
    counts = []
    for path in labels:
        print(path)
        c = count_labels(path)
        print(c)
        counts.append(c)

    # display distrubtion of labels
    ind = np.arange(4)
    width = 0.1
    # hold bargraph info
    rects = []
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    for i in range(len(labels)):
        rects.append(ax.bar(ind, counts[i], width))
        ind = ind + width

    ax.set_xlabel('Attention Levels')
    ax.set_ylabel('Total Instances')
    ax.set_title('Total Instances per Attention Level')
    ax.set_xticks(ind-2*width)
    ax.set_xticklabels(('Low\nAttention', 'Medium\nAttention', 
        'High\nAttention', 'Very High\nAttention'))
#    plt.show()
#
    # save plots
    labels_med, labels_avg, videos = compute_labels(labels)
    median_path = os.path.join(labels_path, 'median_labels.txt')
    average_path = os.path.join(labels_path, 'average_labels.txt')
    save_labels(labels_med, videos, median_path)
    save_labels(labels_avg, videos, average_path)
    med_counts = count_labels(median_path)
    avg_counts = count_labels(average_path)
    
    print(median_path)
    print(med_counts)
    print(average_path)
    print(avg_counts)

    # add distrubtions for median and average
    rects.append(ax.bar(ind+width, med_counts, width))
    rects.append(ax.bar(ind+2*width, avg_counts, width))
    ax.legend((rects[0], rects[1], rects[2], rects[3], rects[4], rects[5],
        rects[6]), ('Subject #1', 'Subject #2', 'Subject #3', 'Subject #4',
            'Subject #5', 'Median', 'Average'))

    plt.show()
    
    fig.savefig('../plots/attention_distributions.png')

if __name__ == '__main__':
    main()
