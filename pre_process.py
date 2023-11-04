from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

# class NLP:
#     def pre_processes(self, received_sentence):
#         sentence = received_sentence 
#         # Tokenization
#         words = word_tokenize(sentence)

#         # Stop-word removal
#         stop_words = set(stopwords.words('english'))
#         filtered_words = [word for word in words if word.lower() not in stop_words]

#         # Stemming
#         stemmer = PorterStemmer()
#         stemmed_words = [stemmer.stem(word) for word in filtered_words]

#         # Sentiment Analysis
#         analyzer = SentimentIntensityAnalyzer()
#         sentiment_scores = analyzer.polarity_scores(sentence)

#         # Return the processed data
#         return stemmed_words, sentiment_scores

#     def polarity_scores(self, received_sentence):
#         textblob_sentiment = TextBlob(received_sentence)
#         return textblob_sentiment.sentiment.polarity

#     def subjectivity_score(self, received_sentence):
#         textblob_sentiment = TextBlob(received_sentence)
#         return textblob_sentiment.sentiment.subjectivity
