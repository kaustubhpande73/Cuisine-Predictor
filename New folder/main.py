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

