from dataset import FailureImageDataset


def main():
    train_dataset = FailureImageDataset(phase='train')
    val_dataset = FailureImageDataset(phase='val')
    test_dataset = FailureImageDataset(phase='test')

    print('Training dataset has %d items' % len(train_dataset))
    print('Validation dataset has %d items' % len(val_dataset))
    print('Test dataset has %d items' % len(test_dataset))


if __name__ == '__main__':
    main()
