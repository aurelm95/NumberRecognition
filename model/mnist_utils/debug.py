if __name__!='__main__':
    print("Este archivo no se puede importar")
    exit()

from PIL import Image
import numpy
import os

import mnist_loader

path_to_data="../data/mnist.pkl.gz"

assert os.path.exists(path_to_data)

training_data, validation_data, test_data = mnist_loader.load_data(path_to_data)



def muestra(data):
    data=data.reshape((28,28))
    img = Image.fromarray(numpy.uint8(data * 255) , 'L')
    img=img.resize( (400, 400))
    return img

img=muestra(training_data[0][1090])
img.show()