import json
import argparse
import os
from sklearn.metrics import classification_report, f1_score
import numpy as np
from collections import Counter

# loads predictions in triplets
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_data(filename):
    # load predictions (danish)
    with open(filename, "r") as lst:
        preds = []
        for line in lst.readlines():
            preds.append(json.loads(line.strip()))
    return preds

def get_f1(chunk_preds, coref_output_file):
    ref, male, fem = [], [], []
    
    labels_ref = list(np.repeat(1, len(chunk_preds)))
    labels_anti_ref = list(np.repeat(0, len(chunk_preds)))

    for p in chunk_preds:
        ref_pred = p[0]
        male_pred = p[1]
        fem_pred = p[2]
        
        if coref_output_file == "xlm-r_coref_da_coref.json":
            cluster_ref = ref_pred['clusters'] #danish
            cluster_male = male_pred['clusters'] #danish
            cluster_fem = fem_pred['clusters'] #danish
        
        else: 
            cluster_ref = ref_pred['predicted_clusters']
            cluster_male = male_pred['predicted_clusters']
            cluster_fem = fem_pred['predicted_clusters']
        
        clusters = [cluster_ref, cluster_male, cluster_fem]   

        for i, cluster in enumerate(clusters):
            if i == 0:
                #print("CLUSTER", cluster)
                if cluster != []:
                    ref.append(1)
                else:
                    ref.append(0)
                
            elif i==1:
                if cluster != []:
                    male.append(1)
                else:
                    male.append(0)

            elif i==2:

                if cluster != []:
                    fem.append(1)
                else:
                    fem.append(0)
                    
    f1_ref = f1_score(labels_ref, ref, zero_division=1, average="weighted")
    f1_male = f1_score(labels_anti_ref, male, zero_division=1, average="weighted")
    f1_fem = f1_score(labels_anti_ref, fem,  zero_division=1, average="weighted")

    f1_fem_ref = f1_score(labels_ref+labels_anti_ref, ref+fem,  zero_division=1, average="weighted")
    f1_male_ref = f1_score(labels_ref+labels_anti_ref, ref+male,  zero_division=1, average="weighted")
    clf_fem_ref = classification_report(labels_ref+labels_anti_ref, ref+fem, zero_division=1)
    clf_male_ref = classification_report(labels_ref+labels_anti_ref, ref+male, zero_division=1)


    clf_report_ref = classification_report(labels_ref, ref, zero_division=1)
    clf_report_anti_male = classification_report(labels_anti_ref, male, zero_division=1)
    clf_report_anti_fem = classification_report(labels_anti_ref, fem, zero_division=1)
    
    print(f'''
    Classification report for the reflexive case:
    F1: {f1_ref}
    {clf_report_ref}
    
    Classification report for anti-reflexive male:
    F1: {f1_male}
    {clf_report_anti_male}
    
    Classification report for anti-reflexive female:
    F1: {f1_fem}
    {clf_report_anti_fem}

    RELATIVE F1 SCORES:
    {f1_fem_ref} for fem+ref
    {clf_fem_ref}

    {f1_male_ref} for male+ref

    {clf_male_ref}

    ''')

def get_percentage(chunk_preds, coref_output_file):

    #reset lists
    ref = 0
    male = 0
    fem = 0
    for p in chunk_preds:
        ref_pred = p[0]
        male_pred = p[1]
        fem_pred = p[2]

        
        if coref_output_file == "xlm-r_coref_da_coref.json":
            cluster_ref = ref_pred['clusters'] #danish
            cluster_male = male_pred['clusters'] #danish
            cluster_fem = fem_pred['clusters'] #danish
        
        else: 
            cluster_ref = ref_pred['predicted_clusters']
            cluster_male = male_pred['predicted_clusters']
            cluster_fem = fem_pred['predicted_clusters']  
        clusters = [cluster_ref, cluster_male, cluster_fem]    

        for i, cluster in enumerate(clusters):
            if i == 0:
                if cluster != []:
                    ref+=1

            elif i==1:
                if cluster != []:
                    male+=1

            elif i==2:
                if cluster != []:
                    fem+=1
    print(f'''
        TOTAL NUMBER OF CLUSTERS PREDICTED BY THE MODEL OUT OF {len(chunk_preds)} SENTENCES:
        Reflexive case: {ref}
        Anti-reflexive - MALE: {male}
        Anti-reflexive - FEMALE: {fem}

        -----------------------------

        PERCENTAGE OF PREDICTED CLUSTERS:

        Correctly predicted clusters in the reflexive cases:
        reflexive: {(ref/len(chunk_preds))*100} % of the sentences are correctly coreferenced

        Wrongly predicted clusters in the anti-reflexive cases:
        anti-reflexive possessive pronoun - MALE: {((male/len(chunk_preds))*100)} % of the sentences are wrongly coreferenced
        anti-reflexive possessive pronoun - FEMALE: {((fem/len(chunk_preds))*100)} % of the sentences are wrongly coreferenced
        
        Relative percentage of predicted clusters - true positives/false positives:
        relative male: {(ref/male)} 
        relative female: {(ref/fem)} 
        (the closer to 1, the more biased is)
        '''
        )

def main(output_type, coref_output_file):
    #get current working directory
    path = os.getcwd()
    #get predictions output from coref model
    filename = os.path.join(path, "outputs", "coref", coref_output_file)
    
    preds = get_data(filename)
    
    if coref_output_file == "xlm-r_coref_da_coref.json":
        chunk_preds = list(chunks(preds[0], 3)) #danish
    else:
        chunk_preds = list(chunks(preds, 3))
    
    if output_type == "f1":
        get_f1(chunk_preds, coref_output_file)

    elif output_type == "percentage":
        get_percentage(chunk_preds, coref_output_file)

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description='Get differences in pronoun prediction')

    parser.add_argument('--coref_output_file', type=str, help='file with coref predictions')
    parser.add_argument('--output', type=str, help='choose between "percentage" or "f1"')
    args = parser.parse_args()
    
    main(args.output, args.coref_output_file)
