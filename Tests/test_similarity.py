

import pytest
import json
import pandas as pd
import spacy
from spacy.lang.en import English
import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
nlp = spacy.load('en_core_web_lg')
import re
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import Project2


def test_similarity():

    data_file = 'yummly.json'
    ingredients = ['tomatoes', 'soy', 'cheese', 'beef', 'salt','ham']
    
    #Parse data file into dataframe
    df = get_data(data_file)

    #Applying test train split to df
    X = df['Ingredients']
    Y = df['Cuisine']
    Z = df['Id']
    X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(X,Y,Z,test_size = 0.3)     

    #Get vectorizer and model
    vector, clf = model(X_train, X_test, Y_train)
#     print(ingredients)  
    print()
    pred, ipr = predict(ingredients, vector, clf)
    
    data = {'Id' : Z_train, 'Cuisine' : Y_train, 'Ingredients' : X_train}
    trn = pd.DataFrame(data)
    
    trn = trn.append(ipr, ignore_index = True)
    trn_vector = vector.fit_transform(tr_df['Ingredients'])
    
    similarity(trn, trn_vector, ingredients, pred)
