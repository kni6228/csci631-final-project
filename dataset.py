import os
import platform
from os.path import abspath, dirname, isdir, join

import cv2
from skimage import io
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from constants import DATASET_PATH, LABELS_PATH


def processData():
    data_path = "data"
    output_path = "data_by_category"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    categories_mappings = {}
    images_in_category = {}
    directory_contents = os.listdir(data_path)
    mapping = 0
    for directory in directory_contents:
        truth_file = os.path.join(data_path, directory)
        truth_file = os.path.join(truth_file, "truth.txt")
        if os.path.exists(truth_file):
            object_label = open(truth_file).read()
            print(object_label)
            object_label = object_label.strip()
            if object_label not in categories_mappings:
                categories_mappings[object_label] = mapping
                mapping += 1
            object_mapping = categories_mappings.get(object_label)
            for file in os.listdir(os.path.join(data_path, directory)):
                if file != "truth.txt":
                    if object_mapping not in images_in_category:
                        images_in_category[object_mapping] = 0
                    else:
                        images_in_category[object_mapping] = images_in_category[object_mapping] + 1

                    image_input_path = os.path.join(data_path, directory)
                    image_input_path = os.path.join(image_input_path, file)
                    # print(image_input_path)
                    image = cv2.imread(image_input_path)
                    # print(image)
                    image_output_path = os.path.join(output_path, str(object_mapping))
                    if not os.path.exists(image_output_path):
                        os.mkdir(image_output_path)
                    cv2.imwrite(os.path.join(image_output_path, str(images_in_category[object_mapping]) + ".jpg"),
                                image)

    file = open(os.path.join(output_path, "labels_mapping.txt"), "w+")
    for key, values in categories_mappings.items():
        file.write(str(values) + ":" + key + "\n")
    file.close()

    return categories_mappings


def generateTrainTestVal(mappings):
    data_path = "data_by_category"
    output_path = "dataset"
    categories_less_images = []
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        os.mkdir(os.path.join(output_path, 'train'))
        os.mkdir(os.path.join(output_path, 'val'))
        os.mkdir(os.path.join(output_path, 'test'))
    new_folder_category = 0
    new_category_mapping = {}

    for category in os.listdir(data_path):
        if category != "labels_mapping.txt":
            file_path = os.path.join(data_path, category)
            image_files = [name for name in os.listdir(file_path)]
            if len(image_files) >= 3:
                train_and_valid, test = train_test_split(image_files, test_size=0.2, random_state=42)
                train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)

                train_dir = os.path.join(output_path, 'train', str(new_folder_category))
                val_dir = os.path.join(output_path, 'val', str(new_folder_category))
                test_dir = os.path.join(output_path, 'test', str(new_folder_category))

                label = None
                for key, value in mappings.items():
                    if str(value) == category:
                        label = key

                new_category_mapping[new_folder_category] = label
                new_folder_category += 1

                if not os.path.exists(train_dir):
                    os.mkdir(train_dir)
                if not os.path.exists(val_dir):
                    os.mkdir(val_dir)
                if not os.path.exists(test_dir):
                    os.mkdir(test_dir)

                for image in train:
                    processImage(image, category, data_path, train_dir)

                for image in test:
                    processImage(image, category, data_path, test_dir)

                for image in val:
                    processImage(image, category, data_path, val_dir)

            else:
                categories_less_images.append(category)

    #print(len(categories_less_images))
    file = open(os.path.join(output_path, "labels_mapping.txt"), "w+")
    for key, values in new_category_mapping.items():
        file.write(str(key) + ":" + str(values) + "\n")
    file.close()


def processImage(file_name, category, input, output):
    image_input_path = os.path.join(input, category)
    image_input_path = os.path.join(image_input_path, file_name)
    # print(image_input_path)
    image = cv2.imread(image_input_path)
    # print(image)
    image_output_path = output
    cv2.imwrite(os.path.join(image_output_path, file_name), image)


def check_path(file_path):
    dir_path = dirname(file_path)
    if not isdir(dir_path):
        print('Creating directory {}'.format(dir_path))
        os.mkdir(dir_path)


class FailureImageDataset(Dataset):
    def __init__(self, phase='train', pretrained_model='densenet'):
        self.dataset_path = abspath(DATASET_PATH.format(phase))
        check_path(self.dataset_path)

        self.count = 0
        self.file_paths = []
        self.process_files()

        self.delimiter = ''
        self.set_delimiter()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if pretrained_model != 'densenet':
            print('Currently only densenet is supported')

        self.label_dict = {}
        self.build_label_dict()

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        img = io.imread(img_path)
        img = self.transform(img)
        label = self.get_label_from_path(img_path)
        return img, label

    def __len__(self):
        return self.count

    def process_files(self):
        for sub_dir in os.listdir(self.dataset_path):
            sub_dir_path = join(self.dataset_path, sub_dir)
            self.count += len(os.listdir(sub_dir_path))
            for file_name in os.listdir(sub_dir_path):
                self.file_paths.append(join(sub_dir_path, file_name))

    def set_delimiter(self):
        if platform.system() == 'Windows':
            self.delimiter = '\\'
        else:
            self.delimiter = '/'

    def get_label_from_path(self, file_path):
        return int(dirname(file_path).split(self.delimiter)[-1])

    def build_label_dict(self):
        for line in open(LABELS_PATH):
            idx, label = line.split(':')
            self.label_dict[idx] = label

    def get_label_dict(self):
        return self.label_dict


if __name__ == '__main__':
    mappings = processData()
    generateTrainTestVal(mappings)
