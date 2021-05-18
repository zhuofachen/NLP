from wordcloud import WordCloud
import matplotlib.pyplot as plt
from cal_similarity import *
from collections import Counter

def make_wc(word_list):
    wordcloud = WordCloud()
    wordcloud.fit_words(dict(Counter(word_list).most_common(40)))

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


if __name__ == '__main__':
    text = load_glove('../dataset/test_content.txt', 'plaintext')
    word_list = preprocess_text(text)
    print(word_list)
    make_wc(word_list)





