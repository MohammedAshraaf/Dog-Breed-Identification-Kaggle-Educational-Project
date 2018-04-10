from PIL import Image
import glob
import imageio
import numpy as np
import csv
import json
from NeuralNetwork import NeuralNetwork


def resize_images(folder_read, folder_save, size=(100, 100)):

    open_training_files = glob.glob('{}/*.jpg'.format(folder_read))
    save_training_files = map(lambda x: x.split('/')[1], open_training_files)
    for file_open, file_save in zip(open_training_files, save_training_files):
        img = Image.open(file_open)
        img = img.resize(size, Image.ANTIALIAS)
        img.save('{}/{}'.format(folder_save, file_save))


def load_image_as_vector(img_path):
    return imageio.imread(img_path).flatten()


def load_images(folder, number_of_images=-1):

    if number_of_images > 0:
        images_files = glob.glob('{}/*.jpg'.format(folder))[:number_of_images]
    else:
        images_files = glob.glob('{}/*.jpg'.format(folder))

    x = np.array(list(map(load_image_as_vector, images_files))) / 255
    y = list(map(lambda file_name: file_name.split('/')[1].split('.')[0], images_files))

    y = convert_classes_to_vector(y, load_ids_to_breed(), load_breeds_to_index())

    return x, y


def load_breeds_to_index():
    breeds = {}
    with open('breeds.csv', 'r', encoding='ascii') as breed_file:
        breed_data = csv.reader(breed_file, delimiter=',', quotechar='|')
        counter = 0
        for breed in breed_data:
            breeds[breed[0]] = counter
            counter += 1

    return breeds


def load_ids_to_breed():
    with open('labels.csv', 'r', encoding='ascii') as labels_file:
        labels = csv.reader(labels_file, delimiter=',')

        dogs = {}
        for label in labels:
            dogs[label[0]] = label[1]

        del dogs['id']
        return dogs


def convert_classes_to_vector(dogs_names, dogs_ids, breeds):
    Y = []
    breeds_length = len(breeds)
    for dog in dogs_names:
        temp = [0 for i in range(breeds_length)]
        temp[breeds[dogs_ids[dog]]] = 1
        Y.append(temp)
    return np.array(Y)


# start with resizing the images to 100 * 100
resize_images('train', 'small')

# load the features X and the targets Y
x, y = load_images('small')
test = int(len(x) * .8)

# divide the data into training and validation
x, x_val = x[: test], x[test:]
y, y_val = y[: test], y[test:]

# initialize the network
network = NeuralNetwork(x.T, y.T, [x.shape[1], 512, 512, 256, y.shape[1]], .01, 2000, True)

# start training
network.train(x_val.T, y_val.T)

# save the best parameters
with open('parameters.json', 'w') as jsonFile:
    json.dump(network.val_param, jsonFile)
