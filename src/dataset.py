import os
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset
import re
import torch

class IdiomDataset(Dataset):
    def __init__(self, dataset_path, labels_vocab, tagger_dict, is_test=False):
        super(IdiomDataset, self).__init__()
        self.dataset_path = dataset_path
        self.labels_vocab = labels_vocab
        self.tagger_dict = tagger_dict
        self.MAX_LEN = 512
        self.is_test = is_test

        self.sentences = self._get_sentences()
        self.encoded_data = []
        self.encode_data()

    def _get_sentences(self):
        """
        A function to read the dataset and return the sentences
        each sentence is a list of dictionaries, where each dictionary contains:
        - token: the token in the sentence
        - tag: the tag of the token
        - lang: the language of the token
        :return: list of sentences
        """
        sentences = []
        sentence = []

        print("Reading dataset...\n")
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            for line in tqdm(f):
                if line.strip():
                    token, tag, lang = line.strip().split("\t")
                    sentence.append({"token":token,"tag":tag,"lang":lang})

                else:
                    if sentence: # prevent empty sentences
                        sentences.append(sentence)
                    sentence = []

            # also after the loop, in case file doesn’t end with a blank line:
            if sentence:
                sentences.append(sentence)

        print("Dataset read.\n")
        print("-" * 50+"\n")
        return sentences


    def encode_data(self):
        
        print("Encoding data...\n")
        # Iterate through each sentence and encode the tokens
        for sentence in tqdm(self.sentences):
            words = []
            labels = []
            langs = []
            for elem in sentence:
                # if word is a word or a punctuation, add it to the list, else append UNK
                if re.search("\w", elem["token"])!=None or re.search("[!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~£−€\¿]+", elem["token"])!=None:
                    words.append(elem["token"])
                else:
                    words.append("UNK")
                labels.append(elem["tag"])
                langs.append(elem["lang"])

            # truncate all three in lock-step 
            words  = words[:self.MAX_LEN]
            labels = labels[:self.MAX_LEN]
            langs  = langs[:self.MAX_LEN]

            if self.is_test:
                # fill labels with empty int
                vectorized_labels = [0] * len(words)
            else :
                vectorized_labels = [self.labels_vocab[label] for label in labels]

            encoded_labels = torch.tensor(vectorized_labels,dtype=torch.long)
        
            lang2id = {"tr":0, "it":1}
            # lang tensor: "tr" → 0, "it" → 1
            lang_ids = [lang2id[l] for l in langs]
            langs_tensor = torch.tensor(lang_ids, dtype=torch.long)
            self.encoded_data.append((words, encoded_labels, langs_tensor))

        print("Data encoded.\n")
        print("-" * 50+"\n")

    def __len__(self):
        return len(self.encoded_data)
 
    def __getitem__(self, idx:int):
        return self.encoded_data[idx]
    