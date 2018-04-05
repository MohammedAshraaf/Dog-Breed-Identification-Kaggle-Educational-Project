from PIL import Image
import glob

for filename in glob.glob('train/*.jpg'):
    img = Image.open(filename)
    filename = filename.split('/')[1].split('.')[0]
    img = img.resize((100, 100), Image.ANTIALIAS)
    img.save('small/'+filename+'.jpg')
