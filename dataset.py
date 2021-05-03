import os
import cv2
from sklearn.model_selection import train_test_split


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
        truth_file = os.path.join(data_path,directory)
        truth_file = os.path.join(truth_file,"truth.txt")
        if os.path.exists(truth_file):
            object_label = open(truth_file).read()
            print(object_label)
            object_label = object_label.strip()
            if object_label not in categories_mappings:
                categories_mappings[object_label] = mapping
                mapping += 1
            object_mapping = categories_mappings.get(object_label)
            for file in os.listdir(os.path.join(data_path,directory)):
                if file != "truth.txt":
                    if object_mapping not in images_in_category:
                        images_in_category[object_mapping] = 0
                    else:
                        images_in_category[object_mapping] = images_in_category[object_mapping] + 1

                    image_input_path = os.path.join(data_path,directory)
                    image_input_path = os.path.join(image_input_path,file)
                    #print(image_input_path)
                    image = cv2.imread(image_input_path)
                    #print(image)
                    image_output_path = os.path.join(output_path,str(object_mapping))
                    if not os.path.exists(image_output_path):
                        os.mkdir(image_output_path)
                    cv2.imwrite(os.path.join(image_output_path, str(images_in_category[object_mapping]) + ".jpg"), image)

    file = open(os.path.join(output_path, "labels_mapping.txt"), "w+")
    for key,values in categories_mappings.items():
        file.write(str(values) +":"+ key+ "\n")
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

    for category in os.listdir(data_path):
        if category != "labels_mapping.txt":
            file_path = os.path.join(data_path, category)
            image_files = [name for name in os.listdir(file_path)]
            if len(image_files)>= 3:
                train_and_valid, test = train_test_split(image_files, test_size=0.2, random_state=42)
                train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)

                train_dir = os.path.join(output_path, 'train', category)
                val_dir = os.path.join(output_path, 'val', category)
                test_dir = os.path.join(output_path, 'test', category)

                if not os.path.exists(train_dir):
                    os.mkdir(train_dir)
                if not os.path.exists(val_dir):
                    os.mkdir(val_dir)
                if not os.path.exists(test_dir):
                    os.mkdir(test_dir)

                for image in train:
                    processImage(image,category,data_path,train_dir)

                for image in test:
                    processImage(image,category,data_path,test_dir)

                for image in val:
                    processImage(image,category,data_path,val_dir)

            else:
               categories_less_images.append(category)

    print(categories_less_images)
    file = open(os.path.join(output_path, "labels_mapping.txt"), "w+")
    for key, values in mappings.items():
        if str(values) not in categories_less_images:
            file.write(str(values) + ":" + key + "\n")
    file.close()


def processImage(file_name,category,input,output):
    image_input_path = os.path.join(input, category)
    image_input_path = os.path.join(image_input_path, file_name)
    # print(image_input_path)
    image = cv2.imread(image_input_path)
    # print(image)
    image_output_path = output
    cv2.imwrite(os.path.join(image_output_path, file_name), image)


if __name__ == '__main__':
    mappings = processData()
    generateTrainTestVal(mappings)
