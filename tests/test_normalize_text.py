# -*- coding: utf-8 -*-
# Example main.py
import argparse
from predict_fun import *


def test_normalize_text():

    inp = '{}/docs/yummly.json'.format(os.path.abspath(os.getcwd()))

    a, b = read_file(inp)

    a = normalize_text(a)

    assert isinstance(a, list) and bool([i.islower() for i in a])