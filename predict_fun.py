import argparse                                   
import os
import json
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity


def read_file(inp):
    # Load the data
    
    with open(inp, "r") as f:
        data = json.load(f)

    # Convert the ingredients into text
    ingredient_id = []
    ingredient_cuisine = []
    ingredient_list = []
    for recipe in data:
        ingredient_id.append(recipe["id"])                                      # capture Ids of cuisines
        ingredient_cuisine.append(recipe["cuisine"])                            # capture cuisine names
        ingredient_list.append(", ".join(recipe["ingredients"]))                 # capture ingredients and convert to a string per cuisine
    #for i in ingredient_list: 
        #print(i+'\n')
    
    return ingredient_list, ingredient_cuisine

def normalize_text(content):
    for i in range(len(content)):
        content[i] = re.sub(r'\d+','', content[i])                            # Removing numbers from the content
        content[i] = re.sub(r'[^\w\s]','', content[i])                        # Removing punctuations from the content like (- , / ? ...)
        content[i] = content[i].strip()                                       # Remove trailing or ending white space characters
        content[i] = content[i].lower()                                       # Convert characters to lower case
    
    return content


def model(normalized_content, cusines, input_string):
    # Transform the text data into a feature matrix
    vectorizer = CountVectorizer()
    vect_ingr_mat = vectorizer.fit_transform(normalized_content)
    vect_str = vectorizer.transform(input_string)
    # Train, Test & Split for applying the model
    X_train, X_test, y_train, y_test = train_test_split(vect_ingr_mat, cusines, test_size=0.25, random_state=42)
    # Train a SVC model
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    res = model.predict(vect_str)
    #print(res)
    score = accuracy_score(y_test, model.predict(X_test))
     
    with open("docs/yummly.json", "r") as f:
        data = json.load(f)
    
    collect = []
    closer = []
    for i in data:
        collect.append(cosine_similarity(vect_str, vectorizer.transform([", ".join(i["ingredients"])]))[0][0])
        closer.append((i["id"], cosine_similarity(vect_str, vectorizer.transform([", ".join(i["ingredients"])]))[0][0]))
    
    return res[0], max(collect), score, closer

def nearest_cusines(closer, n):
    
    sorted_list = sorted(closer, key=lambda x: x[1], reverse= True)

    return sorted_list[1 : (int(n)+1)]
