from PIL import Image
import glob
import imageio


def resize_images(folder_read, folder_save, size=(100, 100)):
    open_training_files = glob.glob('{}/*.jpg'.format(folder_read))
    save_training_files = map(lambda x: x.split('/')[1].split('.')[0], open_training_files)
    for file_open, file_save in zip(open_training_files, save_training_files):
        img = Image.open(file_open)
        img = img.resize(size, Image.ANTIALIAS)
        img.save('{}/{}.jpg'.format(folder_save, file_save))


def load_image_as_vector(img_path):
    return imageio.imread(img_path).flatten()
