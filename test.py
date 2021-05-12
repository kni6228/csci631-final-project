import timeit
from os.path import exists

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from constants import *
from dataset import FailureImageDataset


def test(model, dataloaders, criterion, phase='test'):
    criterion.to(DEVICE)

    model.eval()
    running_loss = 0.0
    running_corrects = 0.0
    test_size = len(dataloaders[phase].dataset)

    start_time = timeit.default_timer()

    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels)

    epoch_loss = running_loss / test_size
    epoch_acc = running_corrects.double() / test_size
    print("[test] Loss: {} Acc: {}".format(epoch_loss, epoch_acc))
    stop_time = timeit.default_timer()
    print("Execution time: " + str(stop_time - start_time) + "\n")


def main():
    if exists(MODEL_PATH):
        model = torch.load(MODEL_PATH)
        dataloaders = {'test': DataLoader(FailureImageDataset(phase='test'), 3, shuffle=True, num_workers=4)}
        criterion = nn.CrossEntropyLoss()
        test(model, dataloaders, criterion)


if __name__ == '__main__':
    main()
