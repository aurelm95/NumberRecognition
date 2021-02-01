import random
import numpy as np
import pickle

class RedNeuronal():
    def __init__(self,neuronas_por_capa):
        # Neuronas por capa es un vector que contiene For example, if the list was [2, 3, 1] then it would be a three-layer network, with the first layer containing 2 neurons, the second layer 3 neurons, and the third layer 1 neuron.
        self.numero_de_capas=len(neuronas_por_capa)
        self.neuronas_por_capa=neuronas_por_capa
        self.umbrales=[np.random.randn(y, 1) for y in neuronas_por_capa[1:]]
        self.pesos = [np.random.randn(y, x) for x, y in zip(neuronas_por_capa[:-1], neuronas_por_capa[1:])]
        # Como podemos ver la primera capa (la de los inputs) no tiene umbrales.
        # np.random.randn(y, x) devuelve un <numpy.ndarray> con "y" filas y "x" columnas
        # De manera que self.umbrales es una lista de matrices columna y self.pesos es una lista de matrices. 
        # la funcion zip(l1,l2) une las listas l1 y l2 en una lista de tuplas

    def sigmoide(self,x):
        # Si "x" es un vector o Numpy.array entonces le aplicara la sigmoide a cada elemento de forma individual.
        return 1.0/(1.0+np.exp(-x))

    def d_sigmoide(self,x):
        return self.sigmoide(x)*(1-self.sigmoide(x))

    def prealimentacion(self,a):
        # prealimentacion = feedforward 
        # retroalimentacion = feedback
        # Esta funcion calcula output de la red nouraonal del input a
        for umbral, peso in zip(self.umbrales, self.pesos):
            a=self.sigmoide(np.dot(peso,a)+umbral)
        return a

    def DescensoGradienteEstocastico(self, datos_entreno, epocas, longitud_mini_lote, indice_aprendizaje, datos_test=None):
        # DGE=Descenso del Gradiente Estocastico (stochastic gradient descent)
        # Parte los "datos_entreno" en mini_lotes y aplica el descenso del gradiente a cada uno de esos mini lotes
        # datos_test son datos que usara para evaluar el aprendizaje tras cada epoca. se podria quitar ya que hara que vaya algo mas lento.
        datos_entreno=list(datos_entreno)
        for epoca in range(epocas): # Usar zip y range para iterar es mucho mas eficiente
            print("epoca:",epoca,'/',epocas)
            random.shuffle(datos_entreno)
            mini_lotes = [datos_entreno[k:k+longitud_mini_lote] for k in range(0, len(datos_entreno), longitud_mini_lote)]
            progreso=0
            for mini_lote in mini_lotes:
                print("progreso:",progreso,'/',len(mini_lotes))
                progreso+=1
                self.aprender_mini_lote(mini_lote, indice_aprendizaje)
            if datos_test:
                print("Epoca "+str(epoca)+': '+str(self.evaluate(datos_test))+'/'+str(len(list(datos_test))))
            print('Epoca '+str(epoca)+' completada\n')

    def aprender_mini_lote(self,mini_lote,indice_aprendizaje):
        # Esta funcion actualizara los valores de los pesos y umbrales tras el entreno del mini_lote
        # si b es un Numpy.array 2x3 entonces b.shape devuelve la tupla (2,3)
        # Siguiendo con este ejemplo tendriamos que np.zeros((2,3)) seria una matriz 2x3 llena de 0, de esta manera inicializamos las matrices
        nabla_umbrales = [np.zeros(umbral.shape) for umbral in self.umbrales]
        nabla_pesos = [np.zeros(peso.shape) for peso in self.pesos]
        for x,y in mini_lote:
            delta_nabla_umbrales, delta_nabla_pesos = self.backprop(x, y) # Calculo el descenso del gradiente para cada elemento del mini-lote
            nabla_umbral = [nu+dnu for nu, dnu in zip(nabla_umbrales, delta_nabla_umbrales)] # Acumulo la suma de gradientes para el umbral
            nabla_pesos = [np+dnp for np, dnp in zip(nabla_pesos, delta_nabla_pesos)] # Acumulo la suma de gradientes para los pesos
        self.pesos = [peso-(indice_aprendizaje/len(mini_lote))*np for peso, np in zip(self.pesos, nabla_pesos)] # Corrijo los pesos dividiendo entre la cantidad de elementos del minilote (porque arriba los acumule sumando simplemente y asi hago la media) multiplicado por el parametro de aprendizaje
        self.umbrales = [umbral-(indice_aprendizaje/len(mini_lote))*nu for umbral, nu in zip(self.umbrales, nabla_umbrales)] # Lo mismo para el umbral

    def backprop(self, x, y):
        """Return a tuple "(nabla_b, nabla_w)" representing the
        gradient for the cost function C_x.  "nabla_b" and
        "nabla_w" are layer-by-layer lists of numpy arrays, similar
        to "self.biases" and "self.weights"."""
        nabla_umbral = [np.zeros(umbral.shape) for umbral in self.umbrales] # Inicializamos estas listas con 0 porque la llevaremos desde el final hasta el principio de manera recursiva (backprop)
        nabla_peso = [np.zeros(peso.shape) for peso in self.pesos]
        # feedforward
        a = [x] # lista para almacenar los valores de las neuronas capa por capa. la activacion  (o valor de las neuronas) de la primera capa es el input de la funcion x
        zs = [] # lista para almecenar las z=w*a+b capa por capa
        for k in range(self.numero_de_capas-1):
            z=np.dot(self.pesos[k], a[k])+self.umbrales[k] # z^(k)=w^(k)*a^(k)+b^(k)
            zs.append(z)
            activacion = self.sigmoide(z)
            a.append(activacion)
        # backward pass
        # Calculo el delta de la ultima capa y los gradientes para umbrales y pesos
        delta = self.cost_derivative(a[-1], y)*self.d_sigmoide(zs[-1]) # delta^(c-1)=(y-s)*f'(z^(c-1))
        nabla_umbral[-1] = delta 
        nabla_peso[-1] = np.dot(delta, a[-2].transpose())# la lista nabla_peso tiene longitud c-1 (no existe delta^(c)) mientras que la lista a tiene longitud c
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.

        # Propagacion hacia atras del error (delta) de manera recursiva
        for l in range(2, self.numero_de_capas):
            try:
                delta = np.dot(self.pesos[-l+1].transpose(), delta) * self.d_sigmoide(zs[-l]) # delta^(k)=w^(k+1)^T*delta^(k+1)
            except:
                delta = np.dot(self.pesos[-l+1], delta) * self.d_sigmoide(zs[-l]) # delta^(k)=w^(k+1)^T*delta^(k+1)
            nabla_umbral[-l] = delta
            try:
                nabla_peso[-l] = np.dot(delta, a[-l-1].transpose())
            except:
                nabla_peso[-l] = np.dot(delta, a[-l-1])
        return (nabla_umbral, nabla_peso)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_data=list(test_data)
        test_results = [(np.argmax(self.prealimentacion(x)), y) for (x, y) in test_data]
        s=sum(int(x == y) for (x, y) in test_results)
        print("Evaluation:",s,'/',len(test_data))
        self.evaluation=float(s/len(test_data))
        return s

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y) 

    def adivina(self, x):
        resultado=self.prealimentacion(x).tolist()
        respuesta=np.argmax(resultado)
        #print 'Estos han sido las activaciones para cada numero:'
        #print resultado
        suma=sum([resultado[r][0] for r in range(10)])
        #print suma
        prob=resultado[respuesta][0]/suma
        print('Creo que es un '+str(respuesta)+' con un '+str(round(prob*100,2))+'% de seguridad')
        return respuesta

    def guardar_red(self,nombre):
        f=open(nombre+'.pickle','wb')
        pickle.dump(self,f)
        f.close()
        print("Red guardada en:",nombre+'.pickle')
