# -*- coding: utf-8 -*-
# Example main.py
import argparse
from predict_fun import *


def test_model():

    input_ingredients = [", ".join(['wheat', 'salt', 'black pepper', 'oil'])]
    
    inp = '{}/docs/yummly.json'.format(os.path.abspath(os.getcwd()))

    a, b = read_file(inp)

    a = normalize_text(a)

    res, c, score, closer = model(a, b, input_ingredients)

    assert (res in b) and isinstance(closer, list) and len(closer) > 0