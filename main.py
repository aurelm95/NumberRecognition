from flask import Flask, render_template, request, make_response
import random
import numpy as np
import base64
import cv2

# Mis imports
from model import model
from model import mnist_to_img
from model import RedNeuronal

app = Flask(
	__name__,
	template_folder='templates',
	static_folder='static'
)
Red=model.cargar_red()

@app.route('/')
@app.route('/draw')
def draw():
	return render_template('draw.html')

@app.route('/paint1')
def paint1():
	return render_template('paint1.html')

@app.route('/predict', methods=['POST'])
def predict():
	if request.method == 'POST':
		final_pred = None
		#Initialize the useless part of the base64 encoded image.
		init_Base64 = 21;
		#Preprocess the image : set the image to 28x28 shape
		#Access the image
		draw = request.form['url']
		print('draw es un',type(draw),'de longitud',len(draw))
		print("empieza asi:",draw[:30])
		#Removing the useless part of the url.
		draw = draw[init_Base64:]
		print('despues de cortar queda asi:',draw[:30])
		print('y se queda con una longitud de',len(draw))
		#Decoding
		draw_decoded = base64.b64decode(draw)
		print('Tras el decode: draw es un',type(draw_decoded),'de longitud',len(draw_decoded))
		image = np.asarray(bytearray(draw_decoded), dtype="uint8")
		print('image es un',type(image),'de longitud',len(image))
		print('image0=',image[0])
		image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
		#Resizing and reshaping to keep the ratio.
		resized = cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA)
		vect = np.asarray(resized, dtype="uint8")
		vect = vect.reshape(1, 1, 28, 28).astype('float32')
		print('vect es un',type(vect),'de longitud',vect.shape)
		print('vect[0] es un',type(vect[0]),'de longitud',vect[0].shape)
		img=mnist_to_img.muestra(vect[0][0])
		img.save('dibujo.png')
		final_pred=np.argmax(Red.prealimentacion(vect[0][0]))


	return render_template('results.html', prediction =final_pred)

if __name__ == "__main__":  
	app.run(
		host='0.0.0.0',  # EStablishes the host, required for repl to detect the site
		port=random.randint(2000, 9000),  # Randomly select the port the machine hosts on.
		debug=True
	)