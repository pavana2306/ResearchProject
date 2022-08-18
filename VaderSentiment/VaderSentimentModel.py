from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def VaderSentiment_Predict(input_sentence):
    sid_obj = SentimentIntensityAnalyzer()
    score = sid_obj.polarity_scores(input_sentence)
    print(f"Predicted Result for the input text {input_sentence} using Vader Sentiment Package is {score['compound']}")
    return score['compound']
