import sys, os
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Get perplexity for pronouns (LM)')
parser.add_argument('--filename', type=str, help='Path to txt file with perplexity scores')

args = parser.parse_args()
filename = args.filename

def main():
    # load txt file into pandas dataframe
    df = pd.read_csv(filename, sep='\t', header=None, names=['all'])

    # extract perpexity loss scores from all collumn
    df['perplexity_male'] = df['all'].str.split(' ').str[-7]  
    df['perplexity_female'] = df['all'].str.split(' ').str[-4] 

    # make into floats
    cols = df.drop(['all'], axis=1).columns
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

    # calculate difference
    df['dif'] = df['perplexity_female'] - df['perplexity_male']

    # return mean dif 
    print(f"Mean difference in perplexity scores: {df['dif'].mean()}")

if __name__:
    main()
