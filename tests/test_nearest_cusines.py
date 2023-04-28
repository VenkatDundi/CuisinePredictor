# -*- coding: utf-8 -*-
# Example main.py
import argparse
from predict_fun import *


def test_model():

    i_file = '/home/gnani/Project_21/docs/yummly.json'

    with open(i_file, "r") as f:
        data = json.load(f)

    ingredient_id = []
    ingredient_cuisine = []
    ingredient_list = []
    for recipe in data:
        ingredient_id.append(recipe["id"])                                      # capture Ids of cuisines
        ingredient_cuisine.append(recipe["cuisine"])                            # capture cuisine names
        ingredient_list.append(", ".join(recipe["ingredients"]))                 # capture ingredients and convert to a string per cuisine
    
    for i in range(len(ingredient_list)):
        ingredient_list[i] = re.sub(r'\d+','', ingredient_list[i])                            # Removing numbers from the ingredient_list
        ingredient_list[i] = re.sub(r'[^\w\s]','', ingredient_list[i])                        # Removing punctuations from the ingredient_list like (- , / ? ...)
        ingredient_list[i] = ingredient_list[i].strip()                                       # Remove trailing or ending white space characters
        ingredient_list[i] = ingredient_list[i].lower()                                       # Convert characters to lower case

    input_ingredients = [", ".join(['wheat', 'salt', 'black pepper', 'oil'])]           # Input Ingredients
    n = 5                                                                               # Number of required predictions
    
    vectorizer = CountVectorizer()
    vect_ingr_mat = vectorizer.fit_transform(ingredient_list)
    vect_str = vectorizer.transform(input_ingredients)
    # Train, Test & Split for applying the model
    X_train, X_test, y_train, y_test = train_test_split(vect_ingr_mat, ingredient_cuisine, test_size=0.25, random_state=42)
    # Train a SVC model
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    res = model.predict(vect_str)
    #print(res)
    score = accuracy_score(y_test, model.predict(X_test))

    
    collect = []
    closer = []
    for i in data:
        collect.append(cosine_similarity(vect_str, vectorizer.transform([", ".join(i["ingredients"])]))[0][0])
        closer.append((i["id"], cosine_similarity(vect_str, vectorizer.transform([", ".join(i["ingredients"])]))[0][0]))
    
    sorted_list = sorted(closer, key=lambda x: x[1], reverse= True)

    a = sorted_list[1 : (int(n)+1)]

    assert len(a) > 0 and isinstance(a, list)