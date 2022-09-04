import os
from PIL import Image
import numpy as np

import mnist_loader

# copiada de mnist_to_img.py
def muestra(data):
    data=data.reshape((28,28))
    img = Image.fromarray(np.uint8(data * 255) , 'L')
    img=img.resize( (400, 400))
    return img

def save_training_images_to_folder(mnist_source_path='../data/mnist.pkl.gz',dest_folder_path='../data/mnist_examples',limit=50):

    assert os.path.exists(dest_folder_path)

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper(mnist_source_path)

    for k,(img, result) in enumerate(list(training_data)[:limit]):
        # print(k,img,result)
        img=muestra(img)
        image_path=os.path.join(dest_folder_path,'image_'+str(k)+'_result_'+str(int(np.argmax(result)))+".jpg")
        print(image_path)
        img.save(image_path)


if __name__=='__main__':
    save_training_images_to_folder()