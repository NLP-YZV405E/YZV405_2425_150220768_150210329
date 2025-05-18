# import stanza
from tqdm import tqdm
import torch
import re
import pandas as pd
import ast
import os
import re

# We implemented these but didnt use them (we were going to show how many of the
# idioms in the test set are also in the training set)

# def initialize(use_gpu=True):
#     print("Initializing Stanza pipelines...")
#     tr_nlp = stanza.Pipeline("tr", processors="tokenize,mwt,lemma", use_gpu=use_gpu, tokenize_pretokenized=True,)
#     it_nlp = stanza.Pipeline("it", processors="tokenize,mwt,lemma", use_gpu=use_gpu, tokenize_pretokenized=True,)
#     print("Stanza pipelines initialized.")
#     print("-" * 50 + "\n")
#     return {"tr": tr_nlp, "it": it_nlp}



# def get_idioms(dataset, tagger_dict):
#     print("Extracting idioms...\n")
#     idioms = []
#     for tokens, tags, langs in dataset:
#         if isinstance(tags, torch.Tensor):
#             tags = tags.tolist()

#         lang_to_idxs = {}
#         for i, lang in enumerate(langs):
#             lang_to_idxs.setdefault(lang, []).append(i)

#         lemmas = [None]*len(tokens)
#         for lang, idxs in lang_to_idxs.items():
#             doc = tagger_dict[lang]([tokens[i] for i in idxs])
#             extracted = [w.lemma for sent in doc.sentences for w in sent.words]
#             assert len(extracted) == len(idxs), "Length mismatch between extracted lemmas and indices"
#             for i, lemma in zip(idxs, extracted):
#                 lemmas[i] = lemma

#         idiom_tokens = [lemma for lemma, tag in zip(lemmas, tags) if tag in (1, 2)]
#         if idiom_tokens:
#             idioms.append("".join(idiom_tokens))
        
#     print("Idioms extracted.")
#     print("-" * 50 + "\n")

#     return idioms




# def overlap_percentage_l1_in_l2(list1, list2):
#     count_in = 0
#     for elem in list1:
#         if elem in list2:
#             count_in += 1
    
#     if len(list1) == 0:
#         return 0.0
#     return count_in/len(list1)

def itu_to_tsv(input_csv, output_tsv):
    df = pd.read_csv(input_csv)
    
    # check if the output directory exists, if not create it
    os.makedirs(os.path.dirname(output_tsv), exist_ok=True)

    with open(output_tsv, 'w', encoding='utf-8') as fout:

        # iterate through each row in the DataFrame
        for _, row in df.iterrows():
            switch = False
            try:
                # Extract the relevant columns
                tokens = ast.literal_eval(row['tokenized_sentence'])
                indices = ast.literal_eval(row['indices'])
                language = row['language']

                # assign O tags to all tokens
                tags = ['O'] * len(tokens)
                # this sentence is not idiomatic
                if indices != [-1]:
                    # assign B-IDIOM tag to the first token then I-IDIOM to the rest
                    for i in indices:
                        if not switch:
                            tags[i] = 'B-IDIOM'
                            switch = True
                        else:
                            tags[i] = 'I-IDIOM'

                # add a space between each sentence
                for token, tag in zip(tokens, tags):
                    fout.write(f"{token}\t{tag}\t{language}\n")
                fout.write("\n")

            # catch errors
            except Exception as e:
                print(f"Error in row: {e}")

def itu_to_tsv_test(input_csv, output_tsv):
    df = pd.read_csv(input_csv)
    
    # check if the output directory exists, if not create it
    os.makedirs(os.path.dirname(output_tsv), exist_ok=True)

    with open(output_tsv, 'w', encoding='utf-8') as fout:
        for _, row in df.iterrows():
            # Extract the relevant columns
            tokens = ast.literal_eval(row['tokenized_sentence'])
            indices = ["" for _ in range(len(tokens))]
            language = row['language']

            # add a space between each sentence
            for token, index in zip(tokens, indices):
                fout.write(f"{token}\t{index}\t{language}\n")
            fout.write("\n")