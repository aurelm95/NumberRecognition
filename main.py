from flask import Flask, render_template, request, make_response
import random
import numpy as np
import base64
import cv2

# Mis imports
from model import trainer
from model import mnist_to_img

# https://www.pythonanywhere.com/forums/topic/13405/
import sys
sys.path.append(r'model/') # necesario para cargar red debido al modulo pickle

app = Flask(
	__name__,
	template_folder='templates',
	static_folder='static'
)
Red=trainer.cargar_red()

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
		#Removing the useless part of the url.
		draw = draw[init_Base64:]
		#Decoding
		draw_decoded = base64.b64decode(draw)
		image = np.asarray(bytearray(draw_decoded), dtype="uint8")
		image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
		#Resizing and reshaping to keep the ratio.
		resized = cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA)
		vect = np.asarray(resized, dtype="uint8")
		vect = vect.reshape(28*28,1).astype('float32')
		print('vect es un',type(vect),'de longitud',vect.shape)

		img=mnist_to_img.muestra(vect)
		img.save('static/dibujo.png')
		
		out=Red.prealimentacion(vect)
		prob=np.round(out*100,decimals=0).tolist()
		final_pred=np.argmax(out)
		print("prediction:",prob,final_pred)

	return render_template('results.html', prediction=final_pred,prob=prob)

@app.after_request
def add_header(r):
	# https://stackoverflow.com/questions/34066804/disabling-caching-in-flask?noredirect=1&lq=1
	# Problema con cache a la hora de cargar la imagen
	"""
	Add headers to both force latest IE rendering engine or Chrome Frame,
	and also to cache the rendered page for 10 minutes.
	"""
	r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
	r.headers["Pragma"] = "no-cache"
	r.headers["Expires"] = "0"
	r.headers['Cache-Control'] = 'public, max-age=0'
	return r

if __name__ == "__main__":  
	app.run(
		host='0.0.0.0',  # EStablishes the host, required for repl to detect the site
		port=random.randint(2000, 9000),  # Randomly select the port the machine hosts on.
		debug=True
	)