import pickle

from model import mnist_loader
from model import RedNeuronal

def entrenar():
	training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
	Red = RedNeuronal.RedNeuronal([784, 30, 10])
	print("empiezo el entreno")
	Red.DescensoGradienteEstocastico(training_data, 3, 10, 3.0, datos_test=test_data)
	print('Red entrenada')
	Red.guardar_red('red_guardada')
	return Red


def cargar_red():
	f = open('model/red_guardada.pickle','rb')
	Red = pickle.load(f)
	f.close()
	return Red




#import img_to_mnist
#Red.adivina(numpy.matrix(img_to_mnist.imageprepare('model/data/5.png')).T)
# imagen a mnist
#https://stackoverflow.com/questions/35842274/convert-own-image-to-mnists-image

# https://trinket.io/python/0bbb6eee2a

# https://techwithtim.net/tutorials/python-module-walk-throughs/turtle-module/drawing-with-mouse/

if __name__=='__main__':
	training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
	Red=cargar_red()
	#Red.evaluate(test_data)

	import mnist_to_img
	import numpy
	l=list(training_data)
	x,y=l[0]
	img=mnist_to_img.muestra(x)
	print("solution:",numpy.argmax(y))
	img.save('ejemplo.png')
	print("prediction:",numpy.argmax(Red.prealimentacion(x)))