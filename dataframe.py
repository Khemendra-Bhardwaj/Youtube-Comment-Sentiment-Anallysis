import pandas as pd
import numpy as np
# from pre_process import NLP
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

sentence_obj = NLP()

df = pd.read_csv('random.csv', encoding='utf-8')
comments = df['comments']
likes = df['likes']

received_polarity = []
received_subjectivity = []
processed_comments = []

for comment in comments:
    processed_comment = sentence_obj.pre_processes(comment) 
    polarity = sentence_obj.polarity_scores(processed_comment)
    subjectivity = sentence_obj.subjectivity_score(processed_comment)
    received_polarity.append(polarity)
    received_subjectivity.append(subjectivity)
    processed_comments.append(processed_comment)






# total_likes = df['likes'].sum()

# df['polarity'] = received_polarity
# df['subjectivity'] = received_subjectivity

# df['weighted_likes'] = [(x) / total_likes for x in likes]
# pd.set_option("display.precision", 10)

# df['weighted_polarity'] = df['weighted_likes'] * df['polarity']
# df['weighted_subjectivity'] = df['weighted_likes'] * df['subjectivity']

# print(df.head(len(likes)))

# net_polarity = df['weighted_polarity'].sum()
# net_subjectivity = df['weighted_subjectivity'].sum()




#  Emoji handling 
#  % positive - negatiuve - neutral % pie chart 
# all negative comments - feedback 
# 