
# coding: utf-8
import warnings
warnings.filterwarnings("ignore")

import pickle
import keras
import re
import numpy as np
import sys
from keras.preprocessing.sequence import pad_sequences

def main(title_orig="No title"):
    model_architect_file=open("model/model.json", "r")
    model_architect=model_architect_file.read()
    model_architect_file.close()
    model=keras.models.model_from_json(model_architect)
    model.load_weights('model/bestModel_weights.h5')
    eligible_words=pickle.load(open("model/eligible_words.p","rb"))
    word_idx_dict=pickle.load(open("model/word_idx_dict.p","rb"))
    score_phrase_dict=pickle.load(open("model/score_phrase_dict.p","rb"))
    score_list=pickle.load(open("model/score_list","rb"))
    regex = re.compile('[^a-zA-Z0-9]')
    title=regex.sub(' ', title_orig).lower().strip().split()
    title=[word_idx_dict[word] if word in eligible_words else 4999 for word in title]
    title=[title]
    title=pad_sequences(title, maxlen=10, value=0)
    score=model.predict(title)[0][0]
    phrase=score_phrase_dict[min(score_list, key=lambda x:abs(x-score))]
    print("'{}' would receive {:0.1f}. It is {} rating.".format(title_orig,score,phrase))
    return


if __name__ == "__main__":
    arg=sys.argv[1] if len(sys.argv)>1 else "No title"
    main(str(arg))
