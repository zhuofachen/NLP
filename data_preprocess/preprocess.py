
"""
several preprocessing steps:
1. remove punctuations, !, *, &, etc.
2. remove emojis,
3, remove step words: the, of, a, and,
4, convert all words to lowercase
5. stemming words, achieved, achieving, achieves, replaced by achieve.
6.


"""


import torchtext
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem.porter import PorterStemmer
import emoji
import string
import pandas
import nltk
import numpy as np



def preprocess_text(text, remove_stop=True, stem_words=True, remove_mentions_hashtags=True):
    """
    eg:
    input: preprocess_text("@water #dream hi hello where are you going be there tomorrow happening happen happens",
    stem_words = True)
    output: ['tomorrow', 'happen', 'go', 'hello']
    """

    # Remove emojis
    print('original', text)
    emoji_pattern = re.compile("[" "\U0001F1E0-\U0001F6FF" "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r"", text)
    text = "".join([x for x in text if x not in emoji.UNICODE_EMOJI])
    if remove_mentions_hashtags:
        text = re.sub(r"@(\w+)", " ", text)
        # text = re.sub(r"#(\w+)", " ", text)
        print('after remove @', text)

    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    print('after remove x', text)
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')  # remove punctuation and numbers
    nopunct = regex.sub(" ", text.lower())
    print('remove punctuation', nopunct)
    words = (''.join(nopunct)).split()

    if (remove_stop):
        words = [w for w in words if w not in ENGLISH_STOP_WORDS]
        # words = [w for w in words if len(w) > 2]  # remove a,an,of etc.

    if (stem_words):
        stemmer = PorterStemmer()
        words = [stemmer.stem(w) for w in words]

    return list(words)


if __name__ == "__main__":
    sentence = "I am the king of the worlding world does do, @@###!%!7*2435243, -_-"
    output = preprocess_text(sentence)
    print(output)