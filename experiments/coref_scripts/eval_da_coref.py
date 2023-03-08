import json
import argparse
import os
from sklearn.metrics import classification_report, f1_score
import numpy as np

# loads predictions in triplets and yield them in chunks of 3
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_data(filename):
    """Load coreference predictions from a file."""
    # load predictions (danish)
    with open(filename, "r") as lst:
        preds = []
        for line in lst.readlines():
            preds.append(json.loads(line.strip()))
    return preds

def get_coref_predictions(chunk_preds, coref_output_file):
    """Get the coreference predictions for the reflexive case and the anti-reflexive cases.
    
    Args:
        - chunk_preds: list of predictions
        - coref_output_file: the name of the file where the predictions are stored
    
    Returns:
        - fem
        - male
        - reflexive
    """
    
    ref, male, fem = [], [], []
    
    # the truth labels for the reflexive case, i.e. all 1s
    labels_ref = list(np.repeat(1, len(chunk_preds)))
    # the truth labels for the anti-reflexive cases, i.e. all 0s
    labels_anti_ref = list(np.repeat(0, len(chunk_preds)))

    for p in chunk_preds:
        ref_pred, male_pred,fem_pred = p[0], p[1], p[2]
        
        if "xlm-r" in coref_output_file:
            predicted_clusters = "clusters"
        else:
            predicted_clusters = "predicted_clusters"

        cluster_ref = ref_pred[predicted_clusters] 
        cluster_male = male_pred[predicted_clusters]
        cluster_fem = fem_pred[predicted_clusters] 
        clusters = [cluster_ref, cluster_male, cluster_fem]   

        for i, cluster in enumerate(clusters):
            if i == 0:
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
    return ref, male, fem, labels_ref, labels_anti_ref

def get_clf_report(chunk_preds, coref_output_file):
    ref, male, fem, labels_ref, labels_anti_ref = get_coref_predictions(chunk_preds, coref_output_file)

    clf_report_ref = classification_report(labels_ref, ref, zero_division=1)
    clf_report_anti_male = classification_report(labels_anti_ref, male, zero_division=1)
    clf_report_anti_fem = classification_report(labels_anti_ref, fem, zero_division=1)

    clf_fem_ref = classification_report(labels_ref+labels_anti_ref, ref+fem, zero_division=1)
    clf_male_ref = classification_report(labels_ref+labels_anti_ref, ref+male, zero_division=1)
    
    print(f'''
    Classification report for the reflexive case:
    {clf_report_ref}
    
    Classification report for anti-reflexive male:
    {clf_report_anti_male}
    
    Classification report for anti-reflexive female:
    {clf_report_anti_fem}

    --------------------------------

    RELATIVE CLASSIFICATION REPORTS:

    FEMALE + REFLEXIVE:
    {clf_fem_ref} for fem+ref
    
    MALE + REFLEXIVE:
    {clf_male_ref} 
    ''')

def get_f1(chunk_preds, coref_output_file):
    '''Get F1 score for the reflexive case and the anti-reflexive cases and print output tables.'''
    
    ref, male, fem, labels_ref, labels_anti_ref = get_coref_predictions(chunk_preds, coref_output_file)
                    
    f1_ref = f1_score(labels_ref, ref, zero_division=1, average="weighted")
    f1_male = f1_score(labels_anti_ref, male, zero_division=1, average="weighted")
    f1_fem = f1_score(labels_anti_ref, fem,  zero_division=1, average="weighted")
       
    f1_fem_ref = f1_score(labels_ref+labels_anti_ref, ref+fem,  zero_division=1, average="weighted")
    f1_male_ref = f1_score(labels_ref+labels_anti_ref, ref+male,  zero_division=1, average="weighted")
    
    print(f'''
    F1 for the reflexive case: {f1_ref}
    F1 for the anti-reflexive male: {f1_male}
    F1 for the anti-reflexive female: {f1_fem}

    --------------------------------

    RELATIVE F1 SCORES:
    
    Female + reflexive: {f1_fem_ref}
    Male + reflexive: {f1_male_ref} 
    ''')

def get_percentage(chunk_preds, coref_output_file):
    '''Get percentage scores for the reflexive case and the anti-reflexive cases and print output tables.'''
    #reset lists
    ref, male, fem = 0, 0, 0
    
    for p in chunk_preds:
        ref_pred, male_pred,fem_pred = p[0], p[1], p[2]
        
        # ensure that the correct key for the predicted clusters is used to index the 
        if "xlm-r" in coref_output_file:
            predicted_clusters = "clusters"
        else:
            predicted_clusters = "predicted_clusters"

        cluster_ref = ref_pred[predicted_clusters] 
        cluster_male = male_pred[predicted_clusters]
        cluster_fem = fem_pred[predicted_clusters] 
        clusters = [cluster_ref, cluster_male, cluster_fem]  

        # count the number of clusters predicted by the model for each case
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
    
    if "xlm-r" in coref_output_file:
        chunk_preds = list(chunks(preds[0], 3)) #danish
    else:
        chunk_preds = list(chunks(preds, 3))
    
    if output_type == "f1":
        get_f1(chunk_preds, coref_output_file)

    elif output_type == "percentage":
        get_percentage(chunk_preds, coref_output_file)

    elif output_type == "clf_report":
        get_clf_report(chunk_preds, coref_output_file)

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description='Get differences in pronoun prediction')

    parser.add_argument('--coref_output_file', type=str, help='file with coref predictions')
    parser.add_argument('--output', type=str, help='choose between "percentage" or "f1" or "clf_report"')
    args = parser.parse_args()
    
    main(args.output, args.coref_output_file)
