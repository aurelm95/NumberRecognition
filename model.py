import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

import RedNeuronal


Red = RedNeuronal.RedNeuronal([784, 30, 10])

Red.DescensoGradienteEstocastico(training_data, 3, 10, 3.0, datos_test=test_data)
print('Red entrenada')
Red.guardar_red('red_guardada')


"""
import pickle
f = open('red_guardada.pickle','rb')
Red = pickle.load(f)
f.close()
"""



import numpy
import img_to_mnist
Red.adivina(numpy.matrix(img_to_mnist.imageprepare('data/5.png')).T)
# imagen a mnist
#https://stackoverflow.com/questions/35842274/convert-own-image-to-mnists-image

# https://trinket.io/python/0bbb6eee2a

# https://techwithtim.net/tutorials/python-module-walk-throughs/turtle-module/drawing-with-mouse/

