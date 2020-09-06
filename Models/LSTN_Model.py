
import pandas as pd
import numpy as np
import string
import re
import nltk
from nltk.corpus import stopwords
import pickle
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten,LSTM
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from numpy import array
from numpy import asarray
from numpy import zeros
import pickle 
from sklearn.externals import joblib 
from os import path

import os
if not path.exists("models"):
    os.mkdir('models')

happyemoticon = r" ([xX;:]-?[dD)]|:-?[\)]|[;:][pP]) "
sademoticon = r" :'?[/|\(] "


def preprocess_text(sen):
    text = remove_tags(sen)
    text = re.sub('[' + string.punctuation + ']', ' ', text)  # remove punctuation
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)  # Single character
    text = re.sub(r'\s+', ' ', text)  # space remove
    text = re.sub(r"([xX;:]-?[dD)]|:-?[\)]|[;:][pP])", happyemoticon, text)
    text = re.sub(r" :'?[/|\(] ", sademoticon, text)
    text = re.sub(r"(.)\1+", r"\1\1", text)
    text = re.sub(r"&\w+;", "", text)
    text = re.sub(r"https?://\S*", "", text)
    text = re.sub(r"https?://\S*", "", text)
    text = re.sub(r"&\w+;", "", text)

    return text


TAG_RE = re.compile(r'<[^>]+>')


def remove_tags(text):
    return TAG_RE.sub('', text)


accuracies={}
scores={}
dataset_names=['A','B','C','D','E','F']

for d in dataset_names:
    data = pd.read_csv(d+".csv",encoding='ISO-8859-1')    
    sentences = data.text
    Y =data.sentiment
    data =0
    
    X = []
    
    for sen in sentences:
        X.append(preprocess_text(sen))
        
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)
    
    X_train = tokenizer.texts_to_sequences(X_train)  ##ONE HOT ENCODING
    X_test = tokenizer.texts_to_sequences(X_test)
    
    vocab_size = len(tokenizer.word_index) + 1
    maxlen = 104
    
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
    
    embeddings_dictionary = dict()
    
    glove_file = open('glove.6B.100d.txt', encoding="utf8")
    
    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = asarray(records[1:], dtype='float32')
        embeddings_dictionary [word] = vector_dimensions
    glove_file.close()
    
    embedding_matrix = zeros((vocab_size, 100))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    
    model = Sequential()
    embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix],mask_zero = True)
    model.add(embedding_layer)
    model.add(LSTM(64,dropout=0.3,recurrent_dropout = 0.3,return_sequences=True))
    model.add(LSTM(64,return_sequences=True))
    model.add(LSTM(64,dropout=0.3,recurrent_dropout = 0.3,return_sequences=True))
    model.add(LSTM(64))
    
    
    
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    LSTM_FIT = model.fit(X_train, y_train, batch_size=64, epochs=4, verbose=1)
    
    score = model.evaluate(X_test, y_test, verbose=1)
    
    save_data = {'classifier':model, 'score':score}
    
    
    
    accuracies.update({d:score[1]})
    scores.update({d:score[0]})

    joblib.dump(model, './models/'+d+'_LSTM_MODEL.pkl')
    joblib.dump(save_data, './models/'+d+'LSTM_with_score_MODELS.pkl')
    
    with open('./models/'+d+'LSTM_tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

print("Test Accuracies LSTM : ",accuracies,"\n")
print("Test Scores LSTM: ",scores,"\n")
