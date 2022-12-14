# Importing libraries
from flask import Flask, render_template, request
import pickle

# Load the Multinomial Naive Bayes model and CountVectorizer object
classifier = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('cv.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = classifier.predict(vect)
    return render_template('index.html', prediction=my_prediction)
   

if __name__ == '__main__':
	app.run(debug=True)