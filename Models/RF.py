
import pickle 
from sklearn.externals import joblib 

import numpy as np
import pandas as pd
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import SVC
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from os import path

import os
if not path.exists("models"):
    os.mkdir('models')


HAPPY_EMO = r" ([xX;:]-?[dD)]|:-?[\)]|[;:][pP]) "
SAD_EMO = r" (:'?[/|\(]) "





def lemmatize_tokenize(text):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in word_tokenize(text)]


    

class TextPreProc(BaseEstimator,TransformerMixin):
    def __init__(self, use_mention=False):
        self.use_mention = use_mention
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        # We can choose between keeping the mentions
        # or deleting them
        if self.use_mention:
            X = X.str.replace(r"@[a-zA-Z0-9_]* ", " @tags ")
        else:
            X = X.str.replace(r"@[a-zA-Z0-9_]* ", "")
            
        # Keeping only the word after the #
        X = X.str.replace("#", "")
        X = X.str.replace(r"[-\.\n]", "")
        # Removing HTML garbage
        X = X.str.replace(r"&\w+;", "")
        # Removing links
        X = X.str.replace(r"https?://\S*", "")
        # replace repeated letters with only two occurences
        # heeeelllloooo => heelloo
        X = X.str.replace(r"(.)\1+", r"\1\1")
        # mark emoticons as happy or sad
        X = X.str.replace(HAPPY_EMO, " happyemoticons ")
        X = X.str.replace(SAD_EMO, " sademoticons ")
        X = X.str.lower()
        X = X.str.replace(r"<[^>]+>","")
        return X


rfaccuracies={}
dataset_names=['A','B','C','D','E','F']

for d in dataset_names:
    print("d",d)
    data = pd.read_csv(d+".csv",encoding='ISO-8859-1')    
    X = data.text
    Y =data.sentiment





    vectorizer = TfidfVectorizer(tokenizer=lemmatize_tokenize,stop_words=stopwords.words('english'))
    pipeline = Pipeline([
        ('text_pre_processing', TextPreProc(use_mention=True)),
        ('vectorizer', vectorizer),])
    
    X = pipeline.fit_transform(X)
        
    learn_data, test_data, sentiments_learning, sentiments_test = train_test_split(X, Y, test_size=0.1,random_state = 43)
    X=0
    Y=0





    classifier=RandomForestClassifier(n_estimators = 10,criterion="entropy",random_state =4)
    main_model=classifier.fit(learn_data, sentiments_learning)
    rfscore = classifier.score(test_data,sentiments_test,sample_weight=None)
    

    rfaccuracies.update({d:rfscore})
    save_data = {'classifier':classifier, 'score':rfscore}
    
    joblib.dump(main_model, './models/'+d+'_RF_MODEL.pkl')
    joblib.dump(save_data, './models/'+d+'_RF_with_score_MODELS.pkl')
  
    
        
print("\n\n\n\n\n\n\n\n Test Accuracies RF : ",rfaccuracies,"\n\n\n\n\n")


