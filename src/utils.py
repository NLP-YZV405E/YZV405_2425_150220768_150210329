# import stanza
from tqdm import tqdm
import torch
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