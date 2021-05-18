import numpy as np
from data_preprocess.preprocess import *
import gensim

def load_glove(filename='dataset/glove.6B/glove.6B.50d.txt', format= 'glove'):

    with open(filename) as f:
        file_content = f.readlines()
    if format == 'plaintext':
        return "".join(file_content)
    else:
        glove_dict = {}
        for line in file_content:
            line_content = line.split()      # the format of glove is ['word', vec];
            glove_dict[line_content[0]] = np.array(line_content[1:], dtype=float)
        return glove_dict

def train_word_vec(filename='../dataset/test_content.txt'):
    with open(filename) as f:
        file_content = f.read()
    words_list = [preprocess_text(file_content)]
    model = gensim.models.Word2Vec(
        words_list,
        size=150,
        window=2,
        min_count=1,
        workers=10,
        iter=10)
    # word_vec_dict = model.wv.vocab
    # print(word_vec_dict['model'])
    # print(model.wv.most_similar('model'), '\n')
    # print(model.wv.word_vec('model'), '\n')
    return model.wv.word_vec


#get centroid of a particular document
def get_centroid(text, gloves):
    words_list = preprocess_text(text)
    word_vec_sum = 0
    words_count = 0
    for w in words_list:
        if w in gloves:
            word_vec_sum += gloves[w]   # sum all the word vectors into one vector, this is like bag of word, not considering sentence meaning
            words_count += 1
    if words_count:
        return word_vec_sum/words_count
    else:
        return 0

#get distance between two centroids
def get_distance (a,b):
    return (np.linalg.norm(a - b))




if __name__ == "__main__":
    glove_dict = load_glove('../dataset/glove.6B/glove.6B.50d.txt', 'glove')
    # glove_dict = train_word_vec()
    text1 = load_glove('../dataset/test_content.txt', 'plaintext')
    text2 = load_glove('../dataset/test_content2.txt', 'plaintext')
    docu_vec1 = get_centroid(text1, glove_dict)
    docu_vec2 = get_centroid(text2, glove_dict)
    print(docu_vec1, docu_vec2)
    dist = get_distance(docu_vec1, docu_vec2)
    print("The distance between two document is : ")
    print(dist)