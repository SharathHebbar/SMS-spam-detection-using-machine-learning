from flask import *
import pickle

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
app = Flask(__name__)

cv = CountVectorizer()


@app.route('/')
def home():
    return render_template('welcome.html')


@app.route('/detect', methods=['POST'])
def detect():
    if request.method == 'POST':
        inp = str(request.form.values())
        res = cv.transform([inp]).toarray()
        clf = pickle.load(open('spam.pickle', 'rb'))
        op = clf.predict(res)

        return render_template('welcome.html', op=op)


if __name__ == '__main__':

    app.run(debug=True)
