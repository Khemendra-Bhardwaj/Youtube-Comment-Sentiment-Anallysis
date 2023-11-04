import pandas as pd
import nltk
from langdetect import detect
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from textblob import TextBlob
from cleantext import clean

import re


# nltk.download("vader_lexicon")
# nltk.download("punkt")
# nltk.download("wordnet")
# nltk.download("stopwords")


df = pd.read_csv("youtube_comments.csv")
english_comments = []
respective_like_count = []
like_counts = []
total_like_count = 0
emojis_score = []

from googletrans import Translator



# out = translator.translate(' आप कैसे हैं ', dest='en')


for index, row in df.iterrows():
    comment = row["comment"]
    like_count = row["like_count"]

    try:
        # considering only english languages 
        if detect(comment) == "en":
            # negation_phrases = identify_negation_phrases(comment)  pending ...
            emojis_score.append(comment)
            english_comments.append( clean(comment, no_emoji=True) )
            respective_like_count.append((like_count+1))
            like_counts.append(like_count)
        else:
            translator=Translator()
            out = translator.translate(comment, dest='en')
            emojis_score.append(comment)
            english_comments.append( clean(comment, no_emoji=True) )
            respective_like_count.append((like_count+1))
            like_counts.append(like_count)


    except Exception:
        pass

# print(len(respective_like_count))
total_like_count = sum(respective_like_count)

for i in range(len(respective_like_count)):
    # print(respective_like_count[i])
    respective_like_count[i] = (respective_like_count[i]/total_like_count)
    # print(respective_like_count)

sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Initialize variables to track most positive and most negative comments
most_positive_comment = None
most_negative_comment = None
max_polarity = -1.0  # Initialize with a low value
min_polarity = 1.0   # Initialize with a high value

# Initialize variables for cumulative polarity and subjectivity
total_polarity = 0.0
total_subjectivity = 0.0

postive_comment_list = []
negative_comment_list = []

total_like_count = sum(respective_like_count);

# -----------------Graph Plotting------------------------- 
polarity_scores = []
subjectivity_scoress=[]
# -----------------Graph Plotting------------------------- 

for i in range(len(english_comments)):
    comment = english_comments[i]
    like_count = respective_like_count[i]
    # Tokenization
    words = word_tokenize(comment)

    # Removing punctuation and converting to lowercase
    words = [word.lower() for word in words if word.isalnum()]

    # Removing stop words
    words = [word for word in words if word not in stop_words]

    # Lemmatization
    words = [lemmatizer.lemmatize(word) for word in words]

    # Calculate polarity and subjectivity using TextBlob
    blob = TextBlob(comment)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    polarity_scores.append(polarity)
    subjectivity_scoress.append(subjectivity) 

    if polarity > 0:
        postive_comment_list.append(comment)
    else: 
        negative_comment_list.append(comment)

    # Update cumulative polarity and subjectivity
    total_polarity += polarity* respective_like_count[i]
    total_subjectivity += subjectivity* respective_like_count[i]

    # Update most positive and most negative commentslang
    if polarity > max_polarity:
        max_polarity = polarity
        most_positive_comment = comment
    if polarity < min_polarity:
        min_polarity = polarity
        most_negative_comment = comment

# Calculate the average polarity and subjectivity
average_polarity = total_polarity 
average_subjectivity = total_subjectivity


import emogis_mapping
from emogis_mapping import f
sentiment_score =0 
for sen in emojis_score:
    sentiment_score += f(sen)

sentiment_score = sentiment_score/len(english_comments)


cummulative_polarity_score = average_polarity
cummulative_subjectivity_score = average_subjectivity 

if sentiment_score !=0:
    cummulative_polarity_score = average_polarity * sentiment_score + average_polarity  
    cummulative_subjectivity_score =  average_subjectivity* sentiment_score + average_subjectivity

final_analysis_polarity= ''
final_analysis_subjectivity = ''



# import matplotlib.pyplot as plt
# from classify.py import subjectivity_scoress, polarity_scores 




# print(out.text)




if cummulative_polarity_score>0.5:
    final_analysis_polarity = 'Almost every person liked the video '
elif cummulative_polarity_score > 0:
    final_analysis_polarity = 'Mix of emotions but net sentiment was in positive sides '
elif cummulative_polarity_score > -0.5:
    final_analysis_polarity = 'Mix of emotions but net sentiment was in negative sides '
else:
    final_analysis_polarity = 'Almost every person disliked the video '


if cummulative_subjectivity_score>0.5:
    final_analysis_subjectivity = 'Almost every person showed their opinion  '
elif cummulative_subjectivity_score > 0:
    final_analysis_subjectivity = 'Mix of factual and biases, most of them were biases  '
elif cummulative_subjectivity_score > -0.5:
    final_analysis_subjectivity = 'Mix of factual and biases, most of them were facts '
else:
    final_analysis_subjectivity = 'Most of them were facts rather them opinion and biases '



from tabulate import tabulate

# Sample data for the table
tt  = len(english_comments)
tp = len(postive_comment_list)
tn = len(negative_comment_list)

if tp < tn:
    tp,tn = tn,tp 



data = [
    ["Most Positive Comment:", most_positive_comment],
    ["Max Polarity:", max_polarity],
    ["Most Negative Comment:", most_negative_comment],
    ["Min Polarity:", min_polarity],
    ["Average Polarity:", average_polarity],
    ["Average Subjectivity:", average_subjectivity],
    ["Emojis Sentiment Score:", sentiment_score],
    ["Cummulative Polarity Score:", cummulative_polarity_score],
    ["Cummulative Subjectivity Score:", cummulative_subjectivity_score],
    ["Considered Comments", tt],
    ["Positive Comments Detected", tp],
    ["Negative Comments Detected", tn]
]


# Generate and display the table
table = tabulate(data, tablefmt="grid")

print(table)
print("Conclusion  \n")
print(final_analysis_polarity + '\n')
print(final_analysis_subjectivity + '\n')

# print("-----Positive Comments--------")

# for i in range(len(postive_comment_list)):
#     print(postive_comment_list[i])
#     print('\n')

# print("-----Negative  Comments--------")

# for i in range(len(negative_comment_list)):
#     print(negative_comment_list[i])
#     print('\n')

# plt.figure(figsize=(8, 6))
# plt.scatter(polarity_scores, subjectivity_scoress, c='blue', alpha=0.5)
# plt.title('Polarity vs. Subjectivity')
# plt.xlabel('Polarity')
# plt.ylabel('Subjectivity')

# # Add grid lines
# plt.grid(True)

# # Display the plot
# plt.show()

import matplotlib.pyplot as plt
import numpy as np



# all_words = " ".join([sentence for sentence in df['clean_comment']])

# wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words)

# # plot the graph
# plt.figure(figsize=(15,8))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.show()


# Generate random polarity scores for demonstration purposes
# num_points = 100
# polarity_scores = [random.uniform(-1, 1) for _ in range(num_points)]
 
# Create a histogram of polarity scores
plt.figure(figsize=(8, 6))
plt.hist(polarity_scores, bins=20, range=(-1, 1), color='blue', alpha=0.5, edgecolor='black')
plt.title('Polarity Scores Frequency')
plt.xlabel('Polarity Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Create a histogram for subjectivity scores
plt.figure(figsize=(8, 6))
plt.hist(subjectivity_scoress, bins=20, range=(-1, 1), color='red', alpha=0.5, edgecolor='black')
plt.title('Subjectivity Scores Frequency')
plt.xlabel('Subjectivity Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

from wordcloud import WordCloud


import matplotlib.pyplot as plt
from collections import Counter

# Sample DataFrame with a "clean_comment" column
# You can replace this with your actual DataFrame
import pandas as pd

# data = {'clean_comment': english_comments}
# df = pd.DataFrame(data)

# # Combine all words from the "clean_comment" column into a single string
# all_words = " ".join([sentence for sentence in df['clean_comment']])

# # Tokenize the combined text into words
# words = all_words.split()

# # Count the frequency of each word
# word_counts = Counter(words)

# # Extract the words and their frequencies
# unique_words = list(word_counts.keys())
# word_frequencies = list(word_counts.values())

# # Create a histogram of word frequencies
# plt.figure(figsize=(10, 6))
# plt.hist(word_frequencies, bins=20, color='blue', alpha=0.5, edgecolor='black')
# plt.title('Word Frequencies')
# plt.xlabel('Word Frequency')
# plt.ylabel('Frequency Count')
# plt.grid(True)
# plt.show()

para  = ""

for i in range(len(english_comments)):
    # tokenized_comment[i] = " ".join(tokenized_comment[i])
    para += english_comments[i] 
    
# df['clean_comment'] = tokenized_comment
# df.head()



from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(para)

# plot the graph
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()