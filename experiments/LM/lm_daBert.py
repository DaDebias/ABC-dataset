from transformers import logging
logging.set_verbosity_warning()

#from transformers import BertTokenizer, BertForMaskedLM
from transformers import AutoTokenizer, AutoModelForPreTraining
import torch
from nltk.tokenize import sent_tokenize
import pandas as pd
import math
import argparse
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser(description='Get perplexity for pronouns (LM)')
parser.add_argument('--lang',  type=str, help='language to evaluate')
parser.add_argument('--filename', type=str, help='file to parse')
parser.add_argument('--data', type=str, default="ABC", help='choose between "ABC" or "benchmark" data. For ABC perplexity is scored per masc, fem and ref\
                    while for a benchmark it is just the average sentence perplexities')

args = parser.parse_args()
lang = args.lang
filename = args.filename
data = args.data
model_name = "danish_bert"

#we use chinese and multilingual bert
if lang  == "zh":
    model = BertForMaskedLM.from_pretrained('bert-base-chinese')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=False)


else:
    #model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased')
    model = AutoModelForPreTraining.from_pretrained('Maltehb/danish-bert-botxo')
    #model = AutoModelForMaskedLM.from_pretrained('xlm-roberta-large')
    #tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
    tokenizer = AutoTokenizer.from_pretrained('Maltehb/danish-bert-botxo', do_lower_case=False)
    #tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large', do_lower_case=False)


model.eval()


def score(sentence, lang):
    #print("here is the sentence before tokenization")
    #print(sentence)
    sentence = "[CLS] "+sentence+" [SEP]"

    if lang == "da":
        prons = ["sin", "sit", "sine"]
    if lang == "ru":
        prons = "свой,своя́,своё,свои́,своего́,свое́й,своего́,свои́х,своему́,свое́й,своему́,\
                свои́м,своего́,свою,своего́,свои́х,свои́м,свое́й,свои́м,свои́ми,своём,свое́й,\
                своём,свои́х,свои,своей,своем,своего,своего,свои".lower().split(",")

    if lang == "sv":
        prons = ["sin", "sitt", "sina"]

    if lang == "zh":
        prons = "自己"

    #print("Tokenizing....")
    tokenize_input = tokenizer.tokenize(sentence)
    segments_ids = [0] * len(tokenize_input)

    segments_tensors = torch.tensor([segments_ids])

    #print("here is the sentence after tokenization")
    #print(tokenize_input)

    no_pron = True
    for i, token in enumerate(tokenize_input):
        #print(token)
        if token in prons:
            #print(f"Found it! this token: {token} was in the pronouns list")
            pron_index = i
            no_pron = False
            break
        else:pass

    if no_pron==True: return "no pronouns to replace"

    #print("masking reflexive pronoun.....")
    #slightly different logics for each language
    tokenize_mask_male = tokenize_input.copy()
    tokenize_mask_female = tokenize_input.copy()
    tokenize_mask_refl = tokenize_input.copy()


    if lang == "da":
        #print("running Danish")
        tokenize_mask_male[pron_index] = "hans"
        tokenize_mask_female[pron_index] = "hendes"

        #print("this is the tokenized sentence after masking with male word")
        #print(tokenize_mask_male)

        #print("this is the tokenized sentence after masking with female word")
        #print(tokenize_mask_female)

    if lang == "ru":
        tokenize_mask_male[pron_index] = "его"
        tokenize_mask_female[pron_index] = "ее"

    if lang == "zh":
        tokenize_mask_male = tokenizer.tokenize(sentence.replace("自己","他 UNK" ))
        tokenize_mask_female = tokenizer.tokenize(sentence.replace("自己","她 UNK" ))
        tokenize_mask_refl = tokenize_input.copy()

        #print(tokenize_mask_female,tokenize_mask_refl )

        truth_index = tokenize_input.index("己")
        male_index = tokenize_mask_male.index("他")
        female_index = tokenize_mask_female.index("她")

    if lang == "sv":
        tokenize_mask_male[pron_index] = "hans"
        tokenize_mask_female[pron_index] = "hennes"

    if lang == "zh":

        tensor_input_male = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_mask_male)])

        tensor_input_female = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_mask_female)])
        tensor_truth = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])

    else:
        tensor_input_male = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_mask_male)])

        tensor_input_female = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_mask_female)])
        tensor_truth = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])


    print("predicting...")

    with torch.no_grad():
        predictions_male = model(tensor_input_male, segments_tensors)[0]
        #print(f"predicions male: {predictions_male}")


    with torch.no_grad():
        predictions_female = model(tensor_input_female, segments_tensors)[0]
        #print(f"predicions female: {predictions_female}")

    with torch.no_grad():
        predictions_truth = model(tensor_truth, segments_tensors)[0]

    #predicted_male = predictions_male[0, pron_index].unsqueeze(0)
    #predicted_female = predictions_female[0, pron_index].unsqueeze(0)
    #truth_ = torch.tensor([tensor_truth[0,pron_index].item()])

    loss_fct = torch.nn.CrossEntropyLoss()
    loss_male = loss_fct(predictions_male.squeeze(),tensor_truth.squeeze()).data
    loss_female = loss_fct(predictions_female.squeeze(),tensor_truth.squeeze()).data
    loss_ref = loss_fct(predictions_truth.squeeze(),tensor_truth.squeeze()).data
    #print(f"loss male: {loss_male}")
    #print(f"loss female: {loss_female}")


    #print(loss)
    return "male: "+ str(loss_male.item())+" "+ str(math.exp(loss_male))+ " female: "+ str(loss_female.item())+ " " +\
            str(math.exp(loss_female)) + " refl: "+ str(loss_ref.item())+ " " + str(math.exp(loss_ref))

def score_standard(sentence):
    sentence = "[CLS] "+sentence+" [SEP]"


    tokenize_input = tokenizer.tokenize(sentence)

    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    predictions=model(tensor_input)
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(predictions.squeeze(),tensor_input.squeeze()).data

    return math.exp(loss)

#read in data and score ABC dataset
if data=='ABC':
    reflexive_sents = []
    with open(filename, "r") as f:
        lines = f.readlines()

        restart = 0
        for line in lines:
            if "--------------" in line: pass
            elif "---" in line:
                restart = 0
            else:
                if restart == 0:
                    reflexive_sents.append(line.strip())
                    restart = 1

    with open("outputs/lm/out_"+lang+"_"+model_name+".txt", "w") as f:
        for i, sent in tqdm(enumerate(reflexive_sents)):
            scores = score(sent, lang)
            f.write(sent +" "+ scores +"\n")

elif data =="benchmark":
    ppl_scores = []
    with open(filename, "r") as f:

        lines = [line.strip() for line in f.readlines()]

        print(len(lines))

        for i, line in tqdm(enumerate(lines)):

            if len(line.split())>0:
                try:
                    ppl = score_standard(line)
                    ppl_scores.append(ppl)

                except Exception:
                    print("error")

    print("perplexity for benchmark: ", np.mean(ppl_scores))
