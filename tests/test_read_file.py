# -*- coding: utf-8 -*-
# Example main.py
import argparse
from predict_fun import *


def test_read_file():

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
    
    a, b = ingredient_list, ingredient_cuisine

    assert isinstance(a, list) and len(a)>0 and isinstance(b, list) and len(b)>0