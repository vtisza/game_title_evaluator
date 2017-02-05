import pickle
import keras
import re
import numpy as np
import sys
from keras.preprocessing.sequence import pad_sequences
import pymongo
from pymongo import MongoClient
import datetime

def predict(title_orig="No title"):
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
    response= "'{}' would score {:0.1f}. It is {} rating.".format(title_orig,score,phrase)
    try:
        mongo_store(title_orig,score,phrase)
    except:
        pass
    return(response)

def mongo_store(title_orig,score,phrase):
    client = MongoClient('mongodb://heroku_app:Heroku2017@ds141209.mlab.com:41209/gametitle')
    db = client.gametitle
    name={'title': title_orig, 'score': str(score), 'phrase': phrase, 'time': str(datetime.datetime.now())}
    db.title.insert_one(name)
    client.close()
