from model.RedNeuronal import RedNeuronal
from model.mnist_utils import img_to_mnist
from model.mnist_utils import mnist_loader
from model.mnist_utils import mnist_to_img

import numpy
import sys
sys.path.append(r'model/') # necesario para cargar red debido al modulo pickle





def entrenar():
	training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
	Red = RedNeuronal([784, 30, 10])
	print("empiezo el entreno")
	Red.DescensoGradienteEstocastico(training_data, 3, 10, 3.0, datos_test=test_data)
	print('Red entrenada')
	Red.guardar_red('red_guardada')
	return Red


if __name__=='__main__':
    Red=RedNeuronal()
    Red.cargar_red()

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    Red.evaluate(test_data)

    l=list(training_data)
    x,y=l[0]
    print("shape:",x.shape)
    img=mnist_to_img.muestra(x)
    img.save("primera.jpg")
    print("solution:",numpy.argmax(y))
    print("prediction:",numpy.argmax(Red.prealimentacion(x)))

    Red.adivina(numpy.matrix(img_to_mnist.imageprepare('model/data/5.png')).T)