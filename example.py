from model import RedNeuronal
from model.mnist_utils import img_to_mnist
from model.mnist_utils import mnist_loader
from model.mnist_utils import mnist_to_img
from model import trainer

import numpy
import sys
sys.path.append(r'model/') # necesario para cargar red debido al modulo pickle

# training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# Red=trainer.cargar_red()
# #Red.evaluate(test_data)


# l=list(training_data)
# x,y=l[0]
# img=mnist_to_img.muestra(x)
# print("solution:",numpy.argmax(y))
# img.save('ejemplo.png')
# print("prediction:",numpy.argmax(Red.prealimentacion(x)))

Red=RedNeuronal.RedNeuronal()
Red.adivina(numpy.matrix(img_to_mnist.imageprepare('model/data/5.png')).T)

def entrenar():
	training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
	Red = RedNeuronal.RedNeuronal([784, 30, 10])
	print("empiezo el entreno")
	Red.DescensoGradienteEstocastico(training_data, 3, 10, 3.0, datos_test=test_data)
	print('Red entrenada')
	Red.guardar_red('red_guardada')
	return Red