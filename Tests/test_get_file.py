



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



data_file = 'yummly.json'

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

    assert df!=None
