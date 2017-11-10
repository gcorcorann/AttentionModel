import cv2
import numpy as np
import os.path
import matplotlib.pyplot as plt

class NearestNeighbor(object):
    
    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train nearest neighbor classifier (i.e. store training examples).

        @param  X:    training data [n_instances x n_features]
        @param  y:    training label [n_instances x 1]
        """
        # the nearest neighbor classifier simply remembers all training data
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        """
        Predict labels on Xte.

        @param  X:    input testing data [n_instances x n_features]

        @return ypred:  predicted labels
        """
        num_test = X.shape[0]
        # let's make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        # loop over all test rows
        for i in range(num_test):
            # find the nearest training image to the i'th test image
            # using the L1 distance (sum of absolute value differences)
            distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
            # get the index with the smallest distance
            min_index = np.argmin(distances)
            # predict the label of the nearest example
            Ypred[i] = self.ytr[min_index]

        return Ypred

def get_frame(video_path, frame_idx):
    """ Grab frame at frame_idx. """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (300, 200))
    return frame

def read_data(labels_path):
    """ Reads data. """
    if os.path.isfile("X.npy") and os.path.isfile("y.npy"):
        X = np.load("X.npy")
        y = np.load("y.npy")
        return X, y
    else:
        X = []
        y = []
        with open(labels_path, "r") as file:
            for line in file:
                print(line)
                video_path, video_label = line.split()
                frame = get_frame(video_path, 50)
                height, width = frame.shape[:2]
                frame = np.reshape(frame, (height*width))
                X.append(frame)
                y.append(video_label)
        X = np.array(X)
        y = np.array(y)
        np.save("X.npy", X)
        np.save("y.npy", y)
        return X, y

def split_data(X, y, split):
    """
    Split data into training and testing.
    
    @param  X:   input data [n_instances x n_features]
    @param  y:   input labels [n_instances x 1]
    @param  split:  ratio of training to testing
                    @pre [0-1]

    @return:    training and testing data
    """
    rand_idx = np.random.permutation(X.shape[0])
    idx = int(X.shape[0] * split)
    Xtr = X[rand_idx[:idx]]
    ytr = y[rand_idx[:idx]]
    Xte = X[rand_idx[idx:]]
    yte = y[rand_idx[idx:]]
    return Xtr, ytr, Xte, yte
    
def main():
    labels_path = "labels_gary.txt"
    X, y = read_data(labels_path)
    Xtr, ytr, Xte, yte = split_data(X, y, 0.6)
    print(Xtr.shape, ytr.shape)
    print(Xte.shape, yte.shape)

    # run nearest neighbor
    nn = NearestNeighbor()
    nn.train(Xtr, ytr)
    ypred = nn.predict(Xte)
    acc = np.mean(ypred == yte)
    acc *= 100
    print("Testing Accuracy:", acc)

    # display image
    plt.figure()
    plt.subplot(121), plt.imshow(Xte[0].reshape(200,300))
    plt.show()

if __name__ == "__main__":
    main()
