from utils.cal_similarity import *
from data_preprocess.preprocess import *
if __name__ == "__main__":
    glove_dict = load_glove()
    print("finished")
    print(glove_dict)
    output = preprocess_text(glove_dict)
    print(output)
