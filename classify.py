import pandas as pd
import nltk
from langdetect import detect
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# nltk.download("vader_lexicon")
# nltk.download("punkt")
# nltk.download("wordnet")
# nltk.download("stopwords")


def remove_emojis(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # Emoticons
                               u"\U0001F300-\U0001F5FF"  # Miscellaneous Symbols and Pictographs
                               u"\U0001F680-\U0001F6FF"  # Transport and Map Symbols
                               u"\U0001F700-\U0001F77F"  # Alphabetic Presentation Forms
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U0001FB00-\U0001FBFF"  # Symbols for Legacy Computing
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)




df = pd.read_csv("youtube_comments.csv")
english_comments = []
like_counts = []

for index, row in df.iterrows():
    comment = row["comment"]
    like_count = row["like_count"]

    try:
        if detect(comment) == "en":
        	# cot = remove_emojis(comment)
        	# print(cot)
            english_comments.append(comment)
            # english_comments.append(comment)
            like_counts.append(like_count)
    except Exception:
        pass


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

for i in range(len(english_comments)):

    comment = english_comments[i]
    like_count = like_counts[i]

    # Tokenization
    words = word_tokenize(comment)

    # Removing punctuation and converting to lowercase
    words = [word.lower() for word in words if word.isalnum()]

    # Removing stop words
    words = [word for word in words if word not in stop_words]

    # Lemmatization
    words = [lemmatizer.lemmatize(word) for word in words]

    # Calculate polarity and subjectivity
    sentiment_score = sia.polarity_scores(comment)
    polarity = sentiment_score["compound"]
    subjectivity = sentiment_score["compound"]

    # Adjust polarity and subjectivity based on like count (in proportion)
    polarity += 0.01 * like_count
    subjectivity += 0.01 * like_count

    # Update cumulative polarity and subjectivity
    total_polarity += polarity
    total_subjectivity += subjectivity

    # Update most positive and most negative comments
    if polarity > max_polarity:
        max_polarity = polarity
        most_positive_comment = comment
    if polarity < min_polarity:
        min_polarity = polarity
        most_negative_comment = comment

# Calculate the average polarity and subjectivity
average_polarity = total_polarity / len(english_comments)
average_subjectivity = total_subjectivity / len(english_comments)

# Print the results
print("Most Positive Comment:")
print(most_positive_comment)
print(f"Max Polarity: {max_polarity}")
print("\nMost Negative Comment:")
print(most_negative_comment)
print(f"Min Polarity: {min_polarity}")
print("\nAverage Polarity:", average_polarity)
print("Average Subjectivity:", average_subjectivity)

