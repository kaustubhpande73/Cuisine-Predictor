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
import pytest


def test_predict():

    data_file = 'yummly.json'
    ingredients = ['eggs', 'soy', 'cheese', 'beef', 'flour','ham']

    #Parse .json file into df
    df = get_data(data_file)

    #Splitting df into 30% test and 70% train dataset 
    X = df['Ingredients']
    Y = df['Cuisine']
    Z = df['Id']
    X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(X,Y,Z,test_size = 0.3)     

    #Get vectorizer and model
    vector, clf = model(X_train, X_test, Y_train)
    #vector = model(X_train, X_test, Y_train)
#     print(ingredients)  
    print()
    #Predict cuisine from user input
    pred, ipr = predict(ingredients, vector, clf)
    
    assert pred!=None

