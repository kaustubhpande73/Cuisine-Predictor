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


def main(ingredients):
    
          
    #Reads the input file (here the json file) and creates a dataframe for id, cuisine and ingredients
    get_data('yummly.json')

    #Split the data into 70-30 porportion for training and testing 
    X = df['Ingredients']
    Y = df['Cuisine']
    Z = df['Id']
    X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(X, Y, Z,test_size = 0.3)     
    
    #Getting model and vectorizing to predict cuisine
    vector, clf = model(X_train, X_test, Y_train)

    #Predict cuisine from user input
    pred, ipr = predict(ingredients, vector, clf)

    
    data = {'Id' : X_train, 'Cuisine' : Y_train, 'Ingredients' : Z_train}
    trn = pd.DataFrame(data)
     
    #User input to train data
    trn = trn.append(ipr, ignore_index = True)
    trn_vector = vector.fit_transform(tr_df['Ingredients'])
    similarity(tr_df, tr_features, ingredients, cuisine)

    
if __name__ == '__main__':

    parser.add_argument('--ingredient', action = 'append', help = 'Enter ingredients')

    args = parser.parse_args()
    if args.ingredient:
        main(args.ingredient)    

