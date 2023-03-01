#This script takes a coref model and runs it on the ABC dataset.
# Duration approx. 20 min ?

from danlp.models import load_xlmr_coref_model
import json
import os
import argparse

parser = argparse.ArgumentParser(description='Get differences in pronoun prediction')
parser.add_argument('--HF_model_name', type=str, help='the name of the HF model to use - for now only "xlm-r_coref"', default="xlm-r_coref")

args = parser.parse_args()

HF_model_name = args.HF_model_name

# load the coreference model
if HF_model_name == "xlm-r_coref": 
    coref_model = load_xlmr_coref_model() 

def main():

    path = os.getcwd()
    print("path:", path)
    # load the data
    with open(os.path.join(path, "data", "COREF_LM", "coref_lm.da"), "r") as f:
        data = f.readlines()

    # remove the ---\n from the data
    data_ = [line_ for line_ in data if line_ != '---\n']
    i = 0
    preds = []

    # run the model on the data
    for i in data_:
        line = [i.split()]
        c = coref_model.predict(line)
        preds.append(c)

    #write predictions to json file
    with open(os.path.join(path, "outputs", "coref", f"{HF_model_name}_da_coref.json"), "w") as final:        
        json.dump(preds, final) 

if __name__ == "__main__":
    main()