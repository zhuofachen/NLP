
import fasttext
import torchtext
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem.porter import PorterStemmer
import emoji
import string
import sys
import nltk




def preprocess_text(text, remove_stop=True, stem_words=True, remove_mentions_hashtags=True):
    """
    several preprocessing steps:
    . remove punctuations, !, *, &, etc.
    . remove emojis,
     convert all words to lowercase
    . Remove numbers.
    . tokenize text.
    . remove step words: the, of, a, and,
    . stemming words, achieved, achieving, achieves, replaced by achieve.
    eg:
    input document, a string, eg. preprocess_text("the doctor is doctoring the  data,  does do, @@###!%!7*2435243, -_-")
    output: word list, e.g. ['doctor', 'doctor', 'data', 'doe']
    """

    # Remove emojis
    # print(text)
    # text = re.sub('[^A-Za-z0-9()!?\]', " ", text)
    #
    # print(text)
    emoji_pattern = re.compile("[" "\U0001F1E0-\U0001F6FF" "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r"", text)
    text = "".join([x for x in text if x not in emoji.UNICODE_EMOJI])
    if remove_mentions_hashtags:
        text = re.sub(r"@(\w+)", " ", text)
        text = re.sub(r"#(\w+)", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub('\[.*?\]', '', text)    # remove any stuff inside [ ]
    text = re.sub('[‘’“”…]', '', text)
    # text = re.sub('[%s]' % re.escape(string.punctuation), '', text)  # remove punctuation
    text = re.sub('\w*\d\w*', '', text)   # remove numbers.


    #remove punctuations
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')  # remove punctuation and numbers
    nopunct = regex.sub(" ", text.lower())  # a string
    # words = (''.join(nopunct)).split()
    words = nopunct.split()   # split into list
    if (remove_stop):
        words = [w for w in words if w not in ENGLISH_STOP_WORDS]
        words = [w for w in words if len(w) >= 2]  # remove a,an,of etc. anything shorter

    if (stem_words):
        stemmer = PorterStemmer()
        words = [stemmer.stem(w) for w in words]

    return list(words)






if __name__ == "__main__":
    sentence = "the doctor is doctoring the  data,  does do, @@###!%!7*2435243, -_-"
    output = preprocess_text(sentence, remove_stop=True, stem_words=True)
    print("testing for sysargs")
    print(output)
    path = nltk.data.find('../dataset/test_content2.txt')
    raw = open(path, 'rU').read()
    print(raw)
    var = sys.argv[1:]
    print(var)