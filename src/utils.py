import spacy
from spacy.cli.download import download as spacy_download

def initialize(language):
    if language == "English": #(1)
        spacy_download("en_core_web_sm")
        spacy_tagger = spacy.load("en_core_web_sm", exclude=["ner", "parser"])

    elif language == "Italian": #(2)
        spacy_download("it_core_news_sm")
        spacy_tagger = spacy.load("it_core_news_sm", exclude=["ner", "parser"])
    elif language == "Turkish": #(3)
        spacy_download("tr_core_news_sm")
        spacy_tagger = spacy.load("tr_core_news_sm", exclude=["ner", "parser"])
    return spacy_tagger


def get_idioms(dataset, spacy_tagger):
    idioms = []

    for elem in dataset:
        idiom = ""
        for token, tag in zip(elem[0], elem[1]):
            if tag == 1: #B tag
                idiom += spacy_tagger(token)[0].lemma_
            elif tag == 2: #I tag
                idiom += spacy_tagger(token)[0].lemma_

        if idiom != "":
            idioms.append(idiom.strip())
    
    return idioms

def overlap_percentage_l1_in_l2(list1, list2):
    count_in = 0
    for elem in list1:
        if elem in list2:
            count_in += 1
    
    return count_in/len(list1)