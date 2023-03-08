#This script takes a coref model and runs it on the ABC dataset.
# Duration approx. 20 min ?

from danlp.models import load_xlmr_coref_model
import json
import os
import argparse

parser = argparse.ArgumentParser(description='Run the danish coreference model on the ABC dataset')
parser.add_argument('--HF_model_name', type=str, help='the name of the HF model to use - for now only "xlm-r_coref"', default="xlm-r_coref")
args = parser.parse_args()

HF_model_name = args.HF_model_name

# load the coreference model
if HF_model_name == "xlm-r_coref": 
    coref_model = load_xlmr_coref_model() 

def main(HF_model_name):

    path = os.getcwd()
    # load the data
    with open(os.path.join(path, "data", "COREF_LM", "coref_lm.da"), "r") as f:
        data = f.readlines()

    # remove the ---\n from the data
    data_ = [line_ for line_ in data if line_ != '---\n']
    i = 0
    preds = []
    clusters = []
    pronouns = ["sin","sit","sine", "hendes", "hans"]
    ghg=0

    # run the model on the data with the predict method and save the predictions
    for i in data_:
        line = [i.split()]
        predicted_clusters = coref_model.predict(line)
        preds.append(predicted_clusters)
        predicted_prons = coref_model.predict_clusters(line)
        if len(predicted_prons) > 0 and predicted_prons[0][1][0] not in pronouns:
            ghg+=1
        clusters.append(predicted_prons)
    print(clusters)
    print(ghg)

    #write predictions to json file
    with open(os.path.join(path, "outputs", "coref", f"{HF_model_name}_da.json"), "w",encoding='utf8') as final:        
        json.dump(preds, final) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the danish coreference model on the ABC dataset')
    parser.add_argument('--HF_model_name', type=str, help='the name of the HF model to use - for now only "xlm-r_coref"', default="xlm-r_coref")
    args = parser.parse_args()

    main(args.HF_model_name)