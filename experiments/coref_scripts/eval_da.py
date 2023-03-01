import json
import argparse
import os

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

def get_f1(chunk_preds):
    pass

def get_percentage(chunk_preds):

    #reset lists
    ref = 0
    male = 0
    fem = 0
    for p in chunk_preds:
        ref_pred = p[0]
        male_pred = p[1]
        fem_pred = p[2]

        cluster_ref = ref_pred['clusters']
        cluster_male = male_pred['clusters']
        cluster_fem = fem_pred['clusters']
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
    print()
    print(f"TOTAL NUMBER OF CLUSTERS PREDICTED BY THE MODEL OUT OF {len(chunk_preds)} SENTENCES:")
    print("Reflexive case", ref)
    print("Anti-reflexive - MALE: ", male)
    print("Anti-reflexive - FEMALE: ", fem)
    print()
    print("-----------------------------")
    print()
    print("PERCENTAGE OF PREDICTED CLUSTERS:")
    print()
    print("Correctly predicted clusters in the reflexive cases:")
    print("reflexive: ", (ref/len(chunk_preds))*100, "% of the sentences are correctly coreferenced")
    print()
    print("Wrongly predicted clusters in the anti-reflexive cases:")
    print("anti-reflexive possessive pronoun - MALE: ", ((male/len(chunk_preds))*100), "% of the sentences are wrongly coreferenced")
    print("anti-reflexive possessive pronoun - FEMALE: ", ((fem/len(chunk_preds))*100), "% of the sentences are wrongly coreferenced")

def main(output_type, coref_output_file):
    #get current working directory
    path = os.getcwd()
    #get predictions output from coref model
    filename = os.path.join(path, "outputs", "coref", args.coref_output_file)
    
    preds = get_data(filename)
    
    chunk_preds = list(chunks(preds[0], 3))
    
    if args.output == "f1":
        get_f1(chunk_preds)

    elif args.output == "percentage":
        get_percentage(chunk_preds)

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description='Get differences in pronoun prediction')

    parser.add_argument('--coref_output_file', type=str, help='file with coref predictions')
    parser.add_argument('--output', type=str, help='choose between "percentage" or "f1"')
    args = parser.parse_args()
    
    main(args.output, args.coref_output_file)