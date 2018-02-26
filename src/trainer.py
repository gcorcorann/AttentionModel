#!/usr/bin/env python3
"""
Training Module.

Author: Gary Corcoran
Date: Jan. 23th, 2018
"""
import time
import torch
from torch.autograd import Variable

def train(network, data_loaders, GPU, num_epochs, criterion, optimizer):
    """
    Train Network.

    @return    training/validation losses, accuracies and best validation
        accuracy
    """
    # start timer
    start = time.time()
    if GPU:
        network = network.cuda()

    # store training and validation statistics
    accuracies = {'Train': [], 'Valid': []}
    losses = {'Train': [], 'Valid': []}
    # store best validation accuracy
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)

        # each epoch has a training and validation phase
        for phase in ['Train', 'Valid']:
            if phase == 'Train':
                network.train(True)  # training mode
            else:
                network.train(False)  # evalulation mode

            running_loss = 0.0
            running_correct = 0
            total_instances = 0

            # iteration over data
            for i_batch, data in enumerate(data_loaders[phase]):
                # get the inputs and labels
                inputs, targets = data['X'], data['y']
                print(inputs.size())
                inputs = inputs[:, 0, :, :]
                print(inputs.size())

                # store in variables
                if GPU:
                    inputs = Variable(inputs.cuda())
                    targets = Variable(targets.cuda())
                else:
                    inputs, targets = Variable(inputs), Variable(targets)

                # zero-out gradient
                network.zero_grad()

                # forward pass (only keep last output)
                outputs = network(inputs)

                # compute loss
                loss = criterion(outputs, targets)
                running_loss += loss.data[0]

                # backward + optimize (only if training phase)
                if phase == 'Train':
                    loss.backward()
                    optimizer.step()

                # compute accuracy
                _, y_pred = torch.max(outputs.data, 1)
                correct = (y_pred == targets.data).sum()
                running_correct += correct
                total_instances += y_pred.size(0)

            # statistics
            epoch_loss = running_loss / total_instances
            epoch_acc = running_correct / total_instances
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            losses[phase].append(epoch_loss)
            accuracies[phase].append(epoch_acc * 100)

            if phase == 'Valid' and epoch_acc > best_acc:
                best_acc = epoch_acc

        print()
    
    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Validation Acc: {:4f}'.format(best_acc))

    return losses, accuracies, best_acc

