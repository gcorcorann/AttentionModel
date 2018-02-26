#!/usr/bin/env python3
"""
Python DataLoader

USAGE:  python data_loader.py
"""
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

class AttentionDataset(Dataset):
    """
    Attention Level Dataset.
    """
    def __init__(self, labels_path, transform=None):
        """
        @param  labels_path:    path to text file with annotations
                                    @pre string
        @param  transform:      optional transform to be applied on image
                                    @pre callable
        """
        # read video paths and labels
        with open(labels_path, 'r') as f:
            data = f.read()
            data = data.split()
            data = np.array(data)
            data = np.reshape(data, (-1,2))
        
        self.data = data
        self.transform = transform
        self.cap = cv2.VideoCapture()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # array for video frames
        X = np.zeros((100, 224, 224, 3), dtype=np.float32)
        vid_path, label = self.data[idx]
        y = int(label) - 1
        # open video
        self.cap.open(vid_path)
        for i in range(100):
            _, frame = self.cap.read()
            
            # apply transform
            if self.transform:
                frame = self.transform(frame)

            # store frame
            X[i] = frame

        # numChannel x Height x Width
        X = np.transpose(X, (0,3,1,2))
        sample = {'X': X, 'y': y}
        # release video capture device
        self.cap.release()
        return sample

class Resize():
    """
    Resizes the image to a given size.

    @param  output_size:    expected output image size
                                @pre tuple
    @return image:         resized image
    """
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, image):
        image = cv2.resize(image, self.output_size)
        return image

class Normalize():
    """
    Rescales image to [0-1], normalizes with across channels with given
    mean and std. and returns image in RGB format.

    @param  mu:     channel means
                        @pre tuple
    @param  std:    channel standard deviation
                        @pre tuple
    @return image:  normalized image
    """
    def __init__(self, mu, std):
        assert isinstance(mu, list)
        assert isinstance(std, list)
        self.mu = mu
        self.std = std

    def __call__(self, image):
        image = image / 255
        b, g, r = cv2.split(image)
        # normalize for ResNet
        b = (b - self.mu[2]) / self.std[2]
        g = (g - self.mu[1]) / self.std[1]
        r = (r - self.mu[0]) / self.std[0]
        # combine into RGB format
        image = cv2.merge((r, g, b))
        return image

def get_loaders(labels_path, batch_size, num_workers):
    """
    Returns torch.utils.data.DataLoader for custom attention dataset.
    """
    composed = transforms.Compose([
        Resize((224,224)),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    attention_dataset = AttentionDataset(labels_path, 
            transform=composed)

    # split dataset into training and validation
    num_instances = len(attention_dataset)
    indices = list(range(num_instances))
    np.random.shuffle(indices)
    split = int(np.floor(num_instances * 0.8))
    
    train_idx, valid_idx = indices[:split][:10], indices[split:][:10]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # Data Loader for Attention Dataset
    # This will return (video, label) for every iteration.
    # video: tensor of shape (batch_size, 3, 224, 224)
    data_loaders = {'Train': DataLoader(attention_dataset, 
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers),
        'Valid': DataLoader(attention_dataset,
            batch_size=batch_size,
            sampler=valid_sampler, 
            num_workers=num_workers)}

    return data_loaders
