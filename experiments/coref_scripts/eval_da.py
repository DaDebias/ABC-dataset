import json
import argparse

parser = argparse.ArgumentParser(description='Get differences in pronoun prediction')
#parser.add_argument('--lang',  type=str, help='language to evaluate')
parser.add_argument('--coref_output', type=str, help='file to parse for translations')
args = parser.parse_args()

#lang = args.lang
filename = args.coref_output
#filename = "/Users/thearolskovsloth/Documents/MASTERS_I_COGSCI/local_cool_prog_thesis/ABC-dataset/outputs/coref/da_coref.json"


# loads predictions in triplets
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
def main():
    # load predictions (danish)
    with open(filename, "r") as lst:
        preds = []
        for line in lst.readlines():
            preds.append(json.loads(line.strip()))

    chunk_preds = list(chunks(preds[0], 3))
    ref = 0
    male = 0
    fem = 0

    chosen = []
    true = []

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


    print("all", len(chunk_preds))
    print("reflexive", ref)
    print("female", fem)
    print("male", male)

    print("Hallucinating clusters...")
    print("reflexive: ", (ref/len(chunk_preds))*100, "% of the time")
    print("male: ", ((male/len(chunk_preds))*100), "% of the time")
    print("female: ", ((fem/len(chunk_preds))*100), "% of the time")

if __name__ == "__main__":
    main()