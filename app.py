from flask import Flask, render_template, request
import pickle
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from numpy import array

import TextBlob.TextBlobModel
from VaderSentiment import VaderSentimentModel

app = Flask(__name__)
app.secret_key = "secret key"

model = load_model("model_weight/BERT_model.h5")

with open('model_weight/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/index', methods=['GET', 'POST'])
def execute():
    if request.method == 'GET':
        input_Text = request.args.get("input_Text")
        data = [input_Text]
        tokenizer.fit_on_texts(data)
        enc = tokenizer.texts_to_sequences(data)
        enc = pad_sequences(enc, maxlen=300, padding='post')

        BERT_Prediction = model.predict(array([enc][0]))[0][0]
        Vader_Prediction = VaderSentimentModel.VaderSentiment_Predict(input_Text)
        TextBlob_Prediction = TextBlob.TextBlobModel.TextBlob_Predict(input_Text)
        overAllScore = (float(BERT_Prediction) + float(Vader_Prediction)) / 2
        if overAllScore >= 0.5:
            result = {
                "prediction": "POS",
                "probability": overAllScore
            }
        else:
            result = {
                "prediction": "NEG",
                "probability": overAllScore
            }
        return result;


if __name__ == '__main__':
    app.run(debug=True)

