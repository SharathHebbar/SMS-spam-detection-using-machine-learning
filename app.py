import streamlit as st
import pickle


class SMSSpamDetection:
    def __init__(self):
        st.header('🦜🔗 SMS Spam Detection')


    def model_response(self, text):
        with open('vectorizer.pickle', 'rb') as f:
            cv = pickle.load(f)
        res = cv.transform([text]).toarray()
        with open('spam.pickle', 'rb') as f:
            clf = pickle.load(f)
        op = clf.predict(res)
        if op[0] == 'ham':
            label = 'not spam'
        else:
            label = 'spam'
        return label

    def get_input(self):
        input_text = st.text_input("Post your message here: ", key='input')
        return input_text

sms = SMSSpamDetection()
user_input = sms.get_input()
response = sms.model_response(user_input)

submit = st.button("Submit")

if submit:
    st.subheader("Output")
    st.write(f"The given message was: {response}")