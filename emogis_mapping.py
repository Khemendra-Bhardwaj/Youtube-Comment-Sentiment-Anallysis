import emoji

emoji_sentiment_scores = {
   
"😃": 0.9,
"😄": 0.9,
"😁": 0.8,
"😆": 0.8,
"😅": 0.7,
"😂": 0.7,
"😇": 0.8,
"😍": 0.9,
"❤️": 0.8,
"🥰": 0.9,
"🤗": 0.8,
"👏": 0.7,
"🙌": 0.7,
"👍": 0.7,
"🌟": 0.8,
"💯": 0.8,
"🔥": 0.7,
"🎉": 0.8,
"🥳": 0.8,
"🤩": 0.8,
"✨": 0.8,
"🌈": 0.8,
"💕": 0.8,
"💖": 0.8,
"💗": 0.7,
"💙": 0.7,
"💚": 0.7,
"💛": 0.7,
"💜": 0.7,
"💝": 0.8,
"💌": 0.7,
"🎁": 0.7,
"🍕": 0.7,
"🍦": 0.7,
"🍭": 0.7,
"🎈": 0.7,
"🎊": 0.7,
"🎆": 0.7,
"🎇": 0.7,
"🏆": 0.7,
"🥇": 0.8,
"🏅": 0.7,
"👑": 0.8,
"🏝️": 0.7,
"🌞": 0.7,
"⛱️": 0.7,
"🚀": 0.7,
"💃": 0.7,
"🕺": 0.7,
"🚲": 0.7,
"😔": -0.7,
"😕": -0.7,
"😞": -0.8,
"😖": -0.8,
"😣": -0.8,
"😢": -0.9,
"😭": -0.9,
"😩": -0.8,
"😫": -0.8,
"😠": -0.9,
"😡": -0.9,
"🤬": -0.9,
"🤯": -0.9,
"🙁": -0.7,
"☹️": -0.7,
"😟": -0.8,
"😨": -0.8,
"😰": -0.8,
"😷": -0.8,
"🤢": -0.8,
"🤮": -0.9,
"🤕": -0.8,
"🤒": -0.8,
"🙏": -0.7,
"👎": -0.7,
"👻": -0.7,
"👽": -0.7,
"💀": -0.9,
"☠️": -0.9,
"🦠": -0.9,
"🤧": -0.7,
"🧐": -0.7,
"🙄": -0.7,
"😴": -0.7,
"💤": -0.7,
"😪": -0.7,
"💩": -0.9,
"🙈": -0.7,
"🙉": -0.7,
"🙊": -0.7,
"💔": -0.9,
"💣": -0.9,
"💥": -0.7,
"🌪️": -0.9,
"🔪": -0.9,
"🧨": -0.9,
"💩": -0.9,
}

import emoji
import regex


def f(text):
    emoji_list = []

    for char in text:
        if char in emoji_sentiment_scores:
            sentiment_score = emoji_sentiment_scores[char]
            emoji_list.append(sentiment_score)
        else:
            # Handle characters not found in the dictionary (you can assign a default value or skip them)
            emoji_list.append(0.0)  # Default value if not found

    # print("Sentiment scores for emojis found in the text:", emoji_list)
    den = len(emoji_list) + 1
    sentiment_score = sum(emoji_list)/ den 
    return sentiment_score
# if sentiment_score > 1:


