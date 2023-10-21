from flask import Flask, render_template, request
import pickle


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    if request.method == 'POST':
        inp = request.form['text_input']

        with open('vectorizer.pickle', 'rb') as f:
            cv = pickle.load(f)
        res = cv.transform([inp]).toarray()
        with open('spam.pickle', 'rb') as f:
            clf = pickle.load(f)
        op = clf.predict(res)
        if op[0] == 'ham':
           label = 'not spam'
        else:
            label = 'spam'
        return render_template('index.html', op=label)


if __name__ == '__main__':

    app.run(debug=True)
