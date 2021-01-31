from flask import Flask, render_template, request, make_response
import random
import numpy as np
import base64

app = Flask(
	__name__,
	template_folder='templates',
	static_folder='static'
)

@app.route('/')
@app.route('/draw')
def draw():
	return render_template('draw.html')

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

	return render_template('results.html', prediction =1)

if __name__ == "__main__":  
	app.run(
		host='0.0.0.0',  # EStablishes the host, required for repl to detect the site
		port=random.randint(2000, 9000),  # Randomly select the port the machine hosts on.
		debug=True
	)