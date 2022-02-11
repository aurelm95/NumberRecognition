from . import mnist_loader


from PIL import Image
import numpy

# https://stackoverflow.com/questions/42353676/display-mnist-image-using-matplotlib
def muestra(data):
    data=data.reshape((28,28))
    img = Image.fromarray(numpy.uint8(data * 255) , 'L')
    img=img.resize( (400, 400))
    return img

if __name__=='__main__':
  training_data, validation_data, test_data = mnist_loader.load_data()
  img=muestra(training_data[0][1090])
  img.show()
