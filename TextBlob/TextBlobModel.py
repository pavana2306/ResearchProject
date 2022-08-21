from textblob import TextBlob


def TextBlob_Predict(input_sentence):
    data = TextBlob(input_sentence)
    result = ''
    if data.sentiment.polarity <= 0:
        result = 'Negative'
    elif data.sentiment.polarity > 0:
        result = 'Positive'

    print(f"Predicted Result for the input text {input_sentence} using Textblob Sentiment Package is {result} with "
          f"the score {data.sentiment.polarity}")
    return data.sentiment.polarity;
