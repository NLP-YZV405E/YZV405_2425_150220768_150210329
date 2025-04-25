import os
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset
import re
import torch

class IdiomDataset(Dataset):
    def __init__(self, dataset_path, tokenizer, labels_vocab, tagger_dict):
        super(IdiomDataset, self).__init__()
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.labels_vocab = labels_vocab
        self.tagger_dict = tagger_dict

        self.sentences = self.get_sentences()
        self.encoded_data = []
        self.encode_data()

    def get_sentences(self):
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
                if line!="\n": 
                    line = line.strip().split("\t")
                    token = line[0]
                    tag = line[1]
                    lang = line[2]
                    elem = {"token": token, "tag":tag, "lang":lang}
                    sentence.append(elem)
                else:
                    sentences.append(sentence)
                    sentence = []
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
            
            vectorized_labels = [self.labels_vocab[label] for label in labels]
            encoded_labels = torch.tensor(vectorized_labels)
            self.encoded_data.append((words, encoded_labels, langs))
        
        print("Data encoded.\n")
        print("-" * 50+"\n")


    def vectorize_words(self, input_vector, special_tokens=True) -> list:
        encoded_words = self.tokenizer.encode(input_vector, add_special_tokens = special_tokens)

        return encoded_words

    def __len__(self):
        return len(self.encoded_data)
 
    def __getitem__(self, idx:int):
        return self.encoded_data[idx]
    