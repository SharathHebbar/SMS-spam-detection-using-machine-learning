from flask import *
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.metrics import accuracy_score

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
        data = pd.read_csv(
            'C:\\projects\ml\\SMS-spam-detection-using-machine-learning\\spam.csv', encoding='latin-1')
        data = data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
        data = data.rename(columns={'v1': 'result'})
        data = data.rename(columns={'v2': 'messages'})
        Y = np.array(data['result'])
        X = np.array(data['messages'])
        x = cv.fit_transform(X)
        best = 0
        for i in range(100):
            xtrain, xtest, ytrain, ytest = train_test_split(x, Y)
            clf = MultinomialNB()
            clf.fit(xtrain, ytrain)
            yp = clf.predict(xtest)
            acc = accuracy_score(ytest, yp)
            if(acc > best):
                best = acc
                with open("spam.pickle", "wb") as f:
                    pickle.dump(clf, f)

        res = cv.transform([inp]).toarray()
        clf = pickle.load(open(
            'C:\\projects\ml\\SMS-spam-detection-using-machine-learning\\spam.pickle', 'rb'))
        op = clf.predict(res)
        # op = op[0]
        # print(op)
        # if op == 'ham':
        #     op = "The message is not spam"
        # else:
        #     op = 'The message is spam'

        return render_template('index.html', op=op)


if __name__ == '__main__':

    app.run(debug=True)
