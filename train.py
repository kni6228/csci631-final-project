import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader

from constants import NUM_CLASSES, DEVICE, FEATURE_EXTRACT, MODEL_PATH
from dataset import FailureImageDataset
from visuals import plot_learning_curve
from test import test


def get_model(num_classes, feature_extracting=True):
    model = models.densenet121(pretrained=True)
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = nn.Linear(1024, num_classes)
    return model


def get_dataloaders(batch_size=3, shuffle=True, num_workers=4):
    train_dataset = FailureImageDataset(phase='train')
    val_dataset = FailureImageDataset(phase='val')
    test_dataset = FailureImageDataset(phase='test')

    print('Training dataset has %d items' % len(train_dataset))
    print('Validation dataset has %d items' % len(val_dataset))
    print('Test dataset has %d items' % len(test_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=shuffle, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=shuffle, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=shuffle, num_workers=num_workers)

    return {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}


def train(model, dataloaders, criterion, optimizer, num_epochs=20):
    since = time.time()
    train_acc_history = []
    val_acc_history = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print('{} loss: {:.4f} acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'train':
                train_acc_history.append(epoch_acc.cpu().numpy())
            else:
                val_acc_history.append(epoch_acc.cpu().numpy())

        print()

    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model, train_acc_history, val_acc_history


def main():
    model = get_model(NUM_CLASSES, FEATURE_EXTRACT).to(DEVICE)
    dataloaders = get_dataloaders()

    criterion = nn.CrossEntropyLoss()
    params_to_learn = model.parameters()
    if FEATURE_EXTRACT:
        params_to_learn = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_learn.append(param)
    optimizer = optim.SGD(params_to_learn, lr=0.1, momentum=0.9)

    model, train_acc_history, val_acc_history = train(model, dataloaders, criterion, optimizer, num_epochs=25)
    torch.save(model, MODEL_PATH)
    plot_learning_curve(train_acc_history, val_acc_history, num_epochs=25)
    test(model, dataloaders, criterion, 'test')


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '8'
    main()
