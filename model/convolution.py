import numpy as np
import time

class Convolucion:

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
        """
        Applies convolution to a given image with current filter, sride and padding

        Image has to be in the form channels first
        
        
        """
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
    
    def aplicar_convolucion_full(self, imagen, debug=False):
        """
        Applies convolution to a given image with the current filter, stride and padding

        Image has to be in the form channels first

        Notes:
        
            - We are using numpy slicing: image[a:b,c:d,e:f]
            - np.sum sums all the elemets os a tensor of any dimension


        
        
        """
        result=[]
        channels, height, width = imagen.shape
        if debug: 
            assert height==width, "ERROR: The image is not squared!"
        tamaño_output=int(np.floor((height+2*self.padding-self.tamaño_filtro)/self.stride)+1) # en realidad deberia servir unicamente como debug para comprobar las dimensiones del output
        print("tamaño_output:",tamaño_output)
        imagen=np.pad(imagen,self.padding)
        # print(imagen)
        for k in range(channels):
            for i in range(tamaño_output):
                fila=[]
                for j in range(tamaño_output):
                    region=imagen[:,i*self.stride:i*self.stride+self.tamaño_filtro,j*self.stride:j*self.stride+self.tamaño_filtro]
                    fila.append(np.sum(region*self.filtro))
                result.append(fila)
        return np.array(result)
    
    

        


    # Sobre implementaciones eficientes de la convolucion de imagenc on filtro
    # https://stackoverflow.com/questions/5710842/fastest-2d-convolution-or-image-filter-in-python


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



t1=np.array([
    [
        [1,2,3],
        [4,5,6],
        [7,8,9]
    ],
    [
        [1,2,3],
        [4,5,6],
        [7,8,9]
    ],
    [
        [1,2,3],
        [4,5,6],
        [7,8,9]
    ]
])

t2=np.array([
    [
        [1,1,1],
        [1,1,1],
        [1,1,1]
    ],
    [
        [1,1,1],
        [1,1,1],
        [1,1,1]
    ],
    [
        [1,1,1],
        [1,1,1],
        [1,1,1]
    ]
])


if __name__=='__main__' and 0:
    c=Convolucion(filtro=igual)
    t0=time.time()
    cuno=c.aplicar_convolucion_ingenua(uno)
    print("timepo:",round(time.time()-t0,4))
    print(cuno)

if __name__=='__main__':
    t=t1*t2
