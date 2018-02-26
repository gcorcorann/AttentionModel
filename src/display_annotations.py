#!/usr/bin/env python3
"""Read Attention Dataset Annotations.
"""
import numpy as np

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
    counts = np.zeros(4)
    with open(labels_path, 'r') as f:
        for line in f:
            line = int(line.split()[1]) - 1
            counts[line] += 1

    return counts

def main():
    """Main Function."""
    labels_gary = '../data/labels_gary.txt'
    labels_allison = '../data/labels_allison.txt'
    labels_alexa = '../data/labels_alexa.txt'
    labels_amanda = '../data/labels_amanda.txt'
    paths = [labels_gary, labels_allison, labels_alexa, labels_amanda]
    classes = ['Low Attention', 'Medium Attention',
            'High Attention', 'Very High Attention']
    print(classes)
    print()

    print("Gary's Labels:   ", count_labels(labels_gary))
    print("Allison's Labels:", count_labels(labels_allison))
    print("Alexa's Labels:  ", count_labels(labels_alexa))
    print("Amanda's Labels: ", count_labels(labels_amanda))
    print()

    labels_med, labels_avg, videos = compute_labels(paths)

#    counts = np.zeros(4)
#    for l in labels_med:
#        idx = int(l-1)
#        counts[idx] += 1
#
#    print("Median Labels:", counts)
#
    counts = np.zeros(4)
    for l in labels_avg:
        idx = int(l-1)
        counts[idx] += 1

    print("Average Labels:", counts)
    print(counts.sum())

#    videos = np.array(videos)
#    videos = np.vstack((videos, labels_med)).T
#    print(videos)
#
#    file = open('annotations_med.txt', 'w')
#
#    for i in videos:
#        s = str(i[0]) + " " + str(i[1][0]) + "\n"
#        print(s)
#        file.write(s)
#
#    file.close()

if __name__ == '__main__':
    main()
