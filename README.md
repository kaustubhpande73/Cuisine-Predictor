# cs5293sp22project2
# Cuisine Predictor
*Project by Kaustubh Pande*

In this project, we were given a dataset of cuisines and ingredients, and we had to make a predictor which on entering ingredients, will predict the cuisine and also give the 5 closest cuisines to the predicted one. 

I have done this project in 4 stages:
1. Reading the json file and loading it in a dataframe
2. Getting the model for the predictor where I used test train split and Multinomial Naive Bayes from sklearn library
3. Predicting the cuisine based on the model and loading in a dataframe
4. Using Cosine Similarity to find the 5 closest recipies to the predicted one

# 1. Reading the file
Here, I have used the 'yummly.json' file for the project. The get_data function reads the json and loads it in a dataframe of the values Id, Cuisine and Ingredients. iT returns the dataframe created.

    def get_data(data_file):
    id, cuisine, ingredients=[], [], [] #creating lists for dataframe
    with open(data_file, 'r') as infile:#opens the data file as infile
          text = infile.read() #reads the data file in 'text'
    #df = pd.read_json('yummly.json')
    data = json.loads(text) #loads the text file in 'data'
    for i in data:
        id.append(i['id'])
        cuisine.append(i['cuisine'])
        ingredients.append(' '.join(i['ingredients']))
    infile.close()
    dat = {'Id' : id, 'Cuisine' : cuisine, 'Ingredients' : ingredients} #Creates a dataframe 
    df = pd.DataFrame(dat)
     #df = pd.json_normalize(data,  meta=['id', 'cuisine', 'ingredients'])
     #df.head(3)

    return df
    
# 2. The Model
Here, I chose to use the train test split from sklearn to train the data. The data was split in to 70% and 30%. The 70% was the train data and the 30% was the test data. To convert the data into features, I have used the Count Vectorizer.

    def model(X_train, X_test, Y_train):
  
    vector= CountVectorizer()
    X_train_v= vector.fit_transform(X_train)
     #print(vector.get_feature_names_out())
     #X_train_v.toarray()
    X_test_v = vector.transform(X_test)
    clf = MultinomialNB()
    clf.fit(X_train_v, Y_train)
    
    return(vector, clf)
    
# 3. Prediction
This function accepts the ingredients and the model to predict the required cuisine on the ingredients.

    def predict(ingredients, vector, model):
    new_ing = []
    new_ing.append(' '.join(ingredients))
    ip = vector.transform(new_ing)
    
    pred = model.predict(ip)
    
    ipr = {'Id' : 'input', 'Cuisine' : pred[0], 'Ingredients' : new_ing[0]}
    print('Cuisine predicted based on the input ingredients: '+ pred[0])
    

    
    return(pred, ipr)
    
# 4. Similarity
Here, I used the Cosine Similarity to find the closest 5 recipies to the predicted cuisine. I reduced the similar_df dataframe to 4000 as I was not getting satisfactory output. Reducing to 4000 gives a reasonable output.

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
        
 # Output 
 The following output was generated:
      
        Input Ingredients: ['tomatoes', 'soy', 'cheese', 'beef', 'salt', 'ham']

        Cuisine Predicted: italian

        5 Closest Recipies: 
        1. Recipe Id: 11231 (0.17025130615174977)
        2. Recipe Id: 4896 (0.18677184190940715)
        3. Recipe Id: 21651 (0.11785113019775795)
        4. Recipe Id: 41346 (0.0468292905790847)
        5. Recipe Id: 17172 (0.0936585811581694)
