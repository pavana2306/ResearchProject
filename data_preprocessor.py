import string
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem.porter import PorterStemmer

class DataFrame_Preprocessor():

    def __init__(self, n_rare_words):
        self.n_rare_words = 10

        print("Preprocessor object created")

    def __remove_punctuation(self, text):

        PUNCT_TO_REMOVE = string.punctuation
        return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

    def __remove_stopwords(self, text):
        STOPWORDS = set(stopwords.words('english'))
        return " ".join([word for word in str(text).split() if word not in STOPWORDS])

    def __stem_words(self, text):
        stemmer = PorterStemmer()
        return " ".join([stemmer.stem(word) for word in text.split()])

    def Text_Preprocessing(self, data):
        data = data[['review', 'rating']]
        data["review"] = data["review"].apply(lambda text: self.__remove_punctuation(text))
        data["review"] = data["review"].apply(lambda text: self.__remove_stopwords(text))
        data["review"] = data["review"].apply(lambda text: self.__stem_words(text))
        data = data.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))
        data['review'] = data['review'].str.replace('\d+', '')
        return data