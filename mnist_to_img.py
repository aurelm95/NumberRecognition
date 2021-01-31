import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data()

from PIL import Image
import numpy

# https://stackoverflow.com/questions/42353676/display-mnist-image-using-matplotlib
def muestra(data):
    data=data.reshape((28,28))
    img = Image.fromarray(numpy.uint8(data * 255) , 'L')
    img=img.resize( (200, 200))
    return img

if __name__=='__main__':
  img=muestra(training_data[0][1090])
  img.show()
