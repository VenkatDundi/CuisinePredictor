# -*- coding: utf-8 -*-
# Example main.py
import argparse
from predict_fun import *


def test_nearest_cusines():

    input_ingredients = [", ".join(['wheat', 'salt', 'black pepper', 'oil'])]
    
    inp = '{}/docs/yummly.json'.format(os.path.abspath(os.getcwd()))

    a, b = read_file(inp)

    a = normalize_text(a)

    res, c, score, closer = model(a, b, input_ingredients)
    
    n = 5

    a = nearest_cusines(closer, n)

    assert len(a) > 0 and isinstance(a, list)