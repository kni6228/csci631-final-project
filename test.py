from constants import *
import torch
import timeit


def test(model, dataloaders, criterion, phase):

    criterion.to(DEVICE)

    model.eval()
    running_loss = 0.0
    running_corrects = 0.0
    result = []
    test_size = len(dataloaders[phase].dataset)

    start_time = timeit.default_timer()

    for inputs,labels in dataloaders[phase]:
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











