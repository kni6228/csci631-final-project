"""
Authors: Karthik Iyer (kni6228@rit.edu), Venkata Thanmai Mande (vm6710@rit.edu)
This file is used for preprocessing the images, and extracting bounding boxes and labels.
"""

import csv
import json
import os

import cv2
import numpy


def images_for_boundingboxDetection():
    original_data_path = "original_data"
    output_dir = "data_bb"
    folders_map = {}

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    directory_contents = os.listdir(original_data_path)
    print(directory_contents)

    with open("data/information.csv") as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            folder = row[0]
            if folder in folders_map:
                folders_map[folder] = folders_map[folder] + 1
            else:
                folders_map[folder] = 1

            if folder in directory_contents:
                object_label = row[9]
                bounding_boxes = row[12]
                converted_bounding_boxes = bounding_boxes.replace("\'", "\"")
                json_bounding_boxes = json.loads("{" + converted_bounding_boxes + "}")
                bounding_boxes = []
                outputPath = None
                for filename, bounding_box in json_bounding_boxes.items():
                    if len(filename.strip()) > 0:
                        imageSaved, outputPathSaved = write_image(original_data_path, output_dir, folder, filename,
                                                                  str(folders_map[folder]))
                        if imageSaved:
                            outputPath = outputPathSaved
                            bounding_box = json.loads(bounding_box)
                            bounding_boxes.append(bounding_box)

                if len(bounding_boxes) > 0:
                    numpy.savetxt(os.path.join(outputPath, "truth" + ".csv"), numpy.asarray(bounding_boxes),
                                  delimiter=",")


def write_image(original_data_path, output_dir, folder, filename, diff):
    image_input_path = os.path.join(original_data_path, folder)
    image_input_path = os.path.join(image_input_path, filename + ".jpg")

    if os.path.exists(image_input_path):
        folder = folder + "_" + diff
        if not os.path.exists(os.path.join(output_dir, folder)):
            os.mkdir(os.path.join(output_dir, folder))
        image = cv2.imread(image_input_path)
        image_output_path = os.path.join(output_dir, folder)
        cv2.imwrite(os.path.join(image_output_path, filename + ".jpg"), image)
        return True, image_output_path

    return False, None


def images_for_objectDetection():
    total_images = 0
    categories = set()

    original_data_path = "original_data"
    output_dir = "data"
    folders_map = {}

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    directory_contents = os.listdir(original_data_path)
    print(directory_contents)

    with open("data/information.csv") as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            folder = row[0]
            if folder in folders_map:
                folders_map[folder] = folders_map[folder] + 1
            else:
                folders_map[folder] = 1

            if folder in directory_contents:
                object_label = row[9]
                bounding_boxes = row[12]
                converted_bounding_boxes = bounding_boxes.replace("\'", "\"")
                json_bounding_boxes = json.loads("{" + converted_bounding_boxes + "}")
                imageSaveSuccessful = False
                outputPath = None
                for filename, bounding_box in json_bounding_boxes.items():
                    if len(filename.strip()) > 0:
                        imageSaved, outputPathSaved = write_image2(original_data_path, output_dir, folder, filename,
                                                                   str(folders_map[folder]), bounding_box)

                        if imageSaved:
                            total_images += 1
                            if object_label not in categories:
                                categories.add(object_label)

                        if not imageSaveSuccessful:
                            imageSaveSuccessful = imageSaved
                            outputPath = outputPathSaved

                if imageSaveSuccessful:
                    file = open(os.path.join(outputPath, "truth" + ".txt"), "w+")
                    file.write(object_label)
                    file.close()

    print(total_images)
    print(categories)
    print(len(categories))
    file = open(os.path.join(output_dir, "labels.txt"), "w+")
    for category in categories:
        file.write(category + "\n")
    file.close()


def write_image2(original_data_path, output_dir, folder, filename, diff, bounding_box):
    try:
        image_input_path = os.path.join(original_data_path, folder)
        image_input_path = os.path.join(image_input_path, filename + ".jpg")

        if os.path.exists(image_input_path):
            folder = folder + "_" + diff
            if not os.path.exists(os.path.join(output_dir, folder)):
                os.mkdir(os.path.join(output_dir, folder))
            image = cv2.imread(image_input_path)

            bounding_box = json.loads(bounding_box)
            x = bounding_box[0]
            y = bounding_box[1]
            width = bounding_box[2]
            height = bounding_box[3]
            image = image[y:y + height, x:x + width]
            image_output_path = os.path.join(output_dir, folder)
            cv2.imwrite(os.path.join(image_output_path, filename + ".jpg"), image)
            return True, image_output_path
        return False, None
    except:
        return False, None


if __name__ == '__main__':
    images_for_objectDetection()
