import numpy as np



def load_glove(filename='dataset/test_content.txt'):
    glove_dict = {}
    with open(filename) as f:
        file_content = f.readlines()
    print(file_content)
    # for line in file_content:
    #     line_content = line.split()
    #     glove_dict[line_content[0]] = np.array(line_content[1:], dtype=float)
    return " ".join(file_content)

#get centroid of a particular document
def get_centroid(text, gloves):
    words_list = preprocess_text(text)
    word_vec_sum = 0
    words_count = 0
    for w in words_list:
        if w in gloves:
            word_vec_sum += gloves[w]
            words_count += 1
    if words_count:
        return word_vec_sum/words_count
    else:
        return 0

#get distance between two centroids
def get_distance (a,b):
    return (np.linalg.norm(a - b))