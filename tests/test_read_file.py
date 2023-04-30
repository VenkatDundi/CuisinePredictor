# -*- coding: utf-8 -*-
# Example main.py
import argparse
from predict_fun import *


def test_read_file():

    inp = '{}/docs/yummly.json'.format(os.path.abspath(os.getcwd()))

    a, b = read_file(inp)

    assert isinstance(a, list) and len(a)>0 and isinstance(b, list) and len(b)>0