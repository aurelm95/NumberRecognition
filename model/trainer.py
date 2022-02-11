import pickle

from .mnist_utils import mnist_loader
from . import RedNeuronal

def entrenar():
	training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
	Red = RedNeuronal.RedNeuronal([784, 30, 10])
	print("empiezo el entreno")
	Red.DescensoGradienteEstocastico(training_data, 3, 10, 3.0, datos_test=test_data)
	print('Red entrenada')
	Red.guardar_red('red_guardada')
	return Red






#import img_to_mnist
#Red.adivina(numpy.matrix(img_to_mnist.imageprepare('model/data/5.png')).T)

if __name__=='__main__':
	training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
	Red=RedNeuronal.RedNeuronal().cargar_red()
	#Red.evaluate(test_data)

	from .mnist_utils import mnist_to_img
	import numpy
	l=list(training_data)
	x,y=l[0]
	img=mnist_to_img.muestra(x)
	print("solution:",numpy.argmax(y))
	img.save('ejemplo.png')
	print("prediction:",numpy.argmax(Red.prealimentacion(x)))