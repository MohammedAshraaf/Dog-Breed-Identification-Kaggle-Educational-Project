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


def resize(folder_read, folder_save, size=(100, 100)):
    open_training = glob.glob('{}/*.jpg'.format(folder_read))
    save_training = {}
    for single_image in open_training:
        try:
            save_training[single_image] = single_image.split('/')[1]
        except:
            continue
    for single_image in save_training:
        img = Image.open (single_image)
        img = img.resize (size, Image.ANTIALIAS)
        img.save ('{}/{}'.format (folder_save, save_training[single_image]))


def load_image_as_vector(img_path):
    return imageio.imread(img_path).flatten()


def load_images(folder, number_of_images=-1, test=False):

    if number_of_images > 0:
        images_files = glob.glob('{}/*.jpg'.format(folder))[:number_of_images]
    else:
        images_files = glob.glob('{}/*.jpg'.format(folder))

    x = np.array(list(map(load_image_as_vector, images_files))) / 255

    y = list(map(lambda file_name: file_name.split('/')[1].split('.')[0], images_files))

    if not test:
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

# write the header of the csv file
with open ('breeds.csv', 'r', encoding='ascii') as breed_file:
    breed_data = csv.reader (breed_file, delimiter=',', quotechar='|')
    f = open('test.csv', 'w')
    f.write('id')
    for breed in breed_data:
        f.write(',%s' % breed[0])
    f.write('\n')

# load training data set x, y for the images
x, y = load_images('small')


divide = int(len(x) * .75)

x, x_val = x[: divide], x[divide:]
y, y_val = y[: divide], y[divide:]

# initialize the network
network = NeuralNetwork(x.T, y.T, [x.shape[1], 300, 300, 200, y.shape[1]], .003, 700, True)

# free up some memory
del x, y

# start training
network.train(x_val.T, y_val.T)

# free up some memory
del x_val, y_val

# load test data set
x, ids = load_images('test64', test=True)

# get the best validation parameters
network.parameters = network.val_param

# calculate the output layer
y, cache = network.deep_model_forward(x.T)

y = y.T

# write the results in csv file
for breed_id, prediction in zip(ids, y):
    s = ','.join(map(str, prediction))
    s = '{},{}\n'.format(breed_id, s)
    f.write(s)
f.close()


