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


#Reads the input file (here the json file) and creates a dataframe for id, cuisine and ingredients
def get_data(data_file):
    id, cuisine, ingredients=[], [], [] #creating lists for dataframe
    with open(data_file, 'r') as infile:#opens the data file as infile
          text = infile.read() #reads the data file in 'text'
#    df = pd.read_json('yummly.json')
    data = json.loads(text) #loads the text file in 'data'
    for i in data:
        id.append(i['id'])
        cuisine.append(i['cuisine'])
        ingredients.append(' '.join(i['ingredients']))
    infile.close()
    dat = {'Id' : id, 'Cuisine' : cuisine, 'Ingredients' : ingredients} #Creates a dataframe 
    df = pd.DataFrame(dat)
#     df = pd.json_normalize(data,  meta=['id', 'cuisine', 'ingredients'])
# #     #df.head(3)

    return df

def model(X_train, X_test, Y_train):
  
    vector= CountVectorizer()
    X_train_v= vector.fit_transform(X_train)
#     print(vector.get_feature_names_out())
#     X_train_v.toarray()
    X_test_v = vector.transform(X_test)
    clf = MultinomialNB()
    clf.fit(X_train_v, Y_train)
    
    return(vector, clf)
  
def predict(ingredients, vector, model):
    new_ing = []
    new_ing.append(' '.join(ingredients))
    ip = vector.transform(new_ing)
    
    pred = model.predict(ip)
    
    ipr = {'Id' : 'input', 'Cuisine' : pred[0], 'Ingredients' : new_ing[0]}
    print('Cuisine predicted based on the input ingredients: '+ pred[0])
    

    
    return(pred, ipr)
  
  
 def similarity(df, doc_mat, ingredients, pred):

    similar = cosine_similarity(doc_mat)
    similar_df = pd.DataFrame(similar)
    
    recipies = df['Id'].values
    recipe_ids = np.where(recipies == 'User_input')[0][0]
    recipe_similarities = similar_df.iloc[recipe_ids].values
    

    print('\nInput Ingredients: ' + str(ingredients) + '\n')
    print('Cuisine Predicted: ' + str(pred[0]) + '\n')

    #Printing 5 closest recipies and the cosine similarity distance
    sim_recipe_ids = np.argsort(-recipe_similarities)[1:6]
    print('5 Closest Recipies: ')
    i = 1
    for k in sim_recipe_ids:
        print(str(i) + '. Recipe Id: ' + str(recipies[k]) + ' (' + str(similar_df[k][4000]) + ')')
        i = i + 1
        
  
