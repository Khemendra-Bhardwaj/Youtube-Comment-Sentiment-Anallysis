negative_prefixes = ["no", "not", "non", "un", "anti", "dis", "mis"]

# Function to identify negation phrases
def identify_negation_phrases(comment):
    words = word_tokenize(comment)
    tagged_words = nltk.pos_tag(words)
    negation_phrases = {'NOA': [], 'NOV': []}

    for i in range(len(tagged_words) - 1):
        if tagged_words[i][0].lower() in negative_prefixes:
            if i + 2 < len(tagged_words) and (
                    tagged_words[i + 2][1].startswith('JJ') or
                    tagged_words[i + 2][1].startswith('VB')
            ):
                negation_phrases['NOA'].append((i, i + 2))
                negation_phrases['NOV'].append((i, i + 2))
            elif i + 4 < len(tagged_words) and (
                    tagged_words[i + 4][1].startswith('JJ') or
                    tagged_words[i + 4][1].startswith('VB')
            ):
                negation_phrases['NOA'].append((i, i + 2, i + 4))
                negation_phrases['NOV'].append((i, i + 2, i + 4))

    return negation_phrases



# def remove_emojis(text):
#     emoji_pattern = re.compile("["
#         u"\U0001F600-\U0001F64F"  # Emoticons
#         u"\U0001F300-\U0001F5FF"  # Miscellaneous Symbols and Pictographs
#         u"\U0001F680-\U0001F6FF"  # Transport and Map Symbols
#         u"\U0001F700-\U0001F77F"  # Alphabetic Presentation Forms
#         u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
#         u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
#         u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
#         u"\U0001FA00-\U0001FA6F"  # Chess Symbols
#         u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
#         u"\U0001FB00-\U0001FBFF"  # Symbols for Legacy Computing
#         "]+", flags=re.UNICODE)
#     return emoji_pattern.sub(r'', text)