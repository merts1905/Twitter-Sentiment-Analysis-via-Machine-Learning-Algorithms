

import pickle 
from sklearn.externals import joblib 
import pandas as pd
import numpy as np
import string
import re
import nltk
from tensorflow.keras import backend
from keras.layers import Flatten

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten,Conv1D
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from numpy import array
from numpy import asarray
from numpy import zeros
from os import path

import os
if not path.exists("models"):
    os.mkdir('models')


happyemoticon = r" ([xX;:]-?[dD)]|:-?[\)]|[;:][pP]) "
sademoticon = r" :'?[/|\(] "




def preprocess_text(sen):
    text =  remove_tags(sen)
    text = re.sub('['+ string.punctuation +']',' ',text)
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"([xX;:]-?[dD)]|:-?[\)]|[;:][pP])",happyemoticon,text)
    text = re.sub(r" :'?[/|\(] ",sademoticon,text)
    text = re.sub(r"(.)\1+", r"\1\1",text)
    text = re.sub(r"&\w+;", "",text)
    text = re.sub(r"https?://\S*", "",text)
    text = re.sub(r"https?://\S*", "",text)
    text = re.sub(r"&\w+;", "",text)
    
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
        
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=40)
    
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)
    
    X_train = tokenizer.texts_to_sequences(X_train)
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
    embedding_layer = Embedding(vocab_size,100,weights = [embedding_matrix],input_length=maxlen,trainable = False)
    model.add(embedding_layer)
    model.add(Conv1D(64, 2, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    

    
    Con_fit = model.fit(X_train, y_train, batch_size=64, epochs=4, verbose=2)
    
    score = model.evaluate(X_test, y_test, verbose=2)
    
    save_data = {'classifier':model, 'score':score}
    
    accuracies.update({d:score[1]})
    scores.update({d:score[0]})

    joblib.dump(Con_fit, './models/'+d+'_CNN_MODEL.pkl')
    
    joblib.dump(save_data, './models/'+d+'_CNN_with_Data_MODELS.pkl')
    
    with open('./models/'+d+'_CNN_tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Test Accuracies CNN : ",accuracies,"\n")
print("Test Scores CNN: ",scores,"\n")

