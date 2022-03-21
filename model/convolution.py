import numpy as np
import time

class Convovolucion:

    def __init__(self,n_filtros=1,tamaño_filtro=3, stride=1, padding=1, filtro=None):
        # constantes
        self.n_filtros=n_filtros
        self.tamaño_filtro=tamaño_filtro
        self.stride=stride
        self.padding=padding

        # variables
        if filtro is None:
            self.filtro=np.random.rand(n_filtros,tamaño_filtro,tamaño_filtro)/(tamaño_filtro*tamaño_filtro) # se divide entre la cantidad para algun tipo de normalizacion
        else:
            self.filtro=np.array(filtro)
            assert self.filtro.shape[0]==self.filtro.shape[1], "El filtro tiene que ser cuadrado y su shape es: "+str(self.filtro.shape)
            self.tamaño_filtro=self.filtro.shape[0]
    
    def regiones_de_imagen(self, imagen):
        self.imagen=imagen
        alto, ancho=imagen.shape
        # for i in range()


    # metodo pensado para tener un unico filtro y una imagen con un unico canal
    def aplicar_convolucion_ingenua(self, imagen):
        result=[]
        # asumo que imagen es una np.array cuadrada de lado .shape[0]
        tamaño_output=int(np.floor((imagen.shape[0]+2*self.padding-self.tamaño_filtro)/self.stride)+1) # en realidad deberia servir unicamente como debug para comprobar las dimensiones del output
        print("tamaño_output:",tamaño_output)
        imagen=np.pad(imagen,self.padding)
        # print(imagen)
        for i in range(tamaño_output):
            fila=[]
            for j in range(tamaño_output):
                region=imagen[i:i+self.tamaño_filtro,j:j+self.tamaño_filtro]
                fila.append(np.sum(region*self.filtro))
            result.append(fila)
        return np.array(result)



uno=np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])

igual=np.array([
    [0,0,0],
    [0,1,0],
    [0,0,0]
])

if __name__=='__main__':
    c=Convovolucion(filtro=igual)
    t0=time.time()
    cuno=c.aplicar_convolucion_ingenua(uno)
    print("timepo:",round(time.time()-t0,4))
    print(cuno)