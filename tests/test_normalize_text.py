# -*- coding: utf-8 -*-
# Example main.py
import argparse
from predict_fun import *


def test_normalize_text():

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

    a = ingredient_list

    assert isinstance(a, list) and bool([i.islower() for i in a])