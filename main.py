import random, string
from flask import Flask, render_template, request, make_response
import telegram_bot
from hora import intervalo,hora_España
import done

app = Flask(  # Create a flask app
	__name__,
	template_folder='templates',  # Name of html file folder
	static_folder='static'  # Name of directory for static files
)

ok_chars = string.ascii_letters + string.digits

@app.route('/')  # What happens when the user visits the site
def predict():
	return 'hola'

if __name__ == "__main__":  # Makes sure this is the main process
	app.run( # Starts the site
		host='0.0.0.0',  # EStablishes the host, required for repl to detect the site
		port=random.randint(2000, 9000),  # Randomly select the port the machine hosts on.
		debug=True
	)