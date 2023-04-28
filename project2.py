# -*- coding: utf-8 -*-
# Example main.py
from predict_fun import *
import argparse                                   
import json
import sys


def main(args):

    if(args.N and len(sys.argv) > 1):                   # Identifies the N from arguments
        n = sys.argv[2]
    #print(n)

    input_ingredients = [", ".join(args.ingredient)]

    inp = '{}/docs/yummly.json'.format(os.path.abspath(os.getcwd()))

    content, cusines = read_file(inp)                               # Read .json file
    
    normalized_content = normalize_text(content)        # Normalizing the text - Ingredient's string content

    #normalized_content.append(", ".join(args.ingredient))

    cusine_name, cusine_score, accuracy, closer = model(normalized_content, cusines, input_ingredients)

    result = nearest_cusines(closer, n)

    json_format = {}

    json_format["cuisine"] = cusine_name
    json_format["score"] = round(cusine_score, 5)
    json_format["closest"] = [{"id" : i[0], "score" : round(i[1], 5)} for i in result]

    print(json.dumps(json_format, indent=4))
    
   
if __name__ == '__main__':                  
    
    parser = argparse.ArgumentParser(description='Provide some Ingredients!')   # Argument Parser
    parser.add_argument('--N', type=str, help='Number of suggestions to be made..!', required=True)     # Specifying filters to be detected in arguments
    parser.add_argument('--ingredient', action='append', help="Ingredients", required=True)
    args = parser.parse_args()
    
    if args.N and args.ingredient:                  # Validating if --input exists
        main(args)
    else:
        print("Please specify number of suggestions --N flag & Ingredients --ingredient flag!")         #Error message on missing --N & --ingredient filter




    