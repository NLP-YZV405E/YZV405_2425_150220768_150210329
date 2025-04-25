import random
import os
import numpy as np
import torch
from transformers import BertConfig, BertTokenizer, BertModel
from utils import *
from dataset import IdiomDataset


if __name__=="__main__":

    SEED = 2
    # set seeds to get reproducible results
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    # gpuda bazen randomluk olabiliyormuş onu kaldırmak için
    torch.backends.cudnn.deterministic = True

    # create bert
    model_name = 'bert-base-multilingual-cased'
    # output hidden states -> it helps to get hidden states from bert
    bert_config = BertConfig.from_pretrained(model_name, output_hidden_states=True)
    bert_tokenizer = BertTokenizer.from_pretrained(model_name)
    # get bert weights
    bert_model = BertModel.from_pretrained(model_name, config=bert_config)

    # train, update or test mode selection
    mode = input("Do you want to train or test the model? (train, update, test): ").strip().lower()
    assert mode in ['train', 'update', 'test'], "Mode must be one of train, update, test"
    # select the dataset
    dataset_selection = input("Select the dataset (ID10M, ITU, PARSEME): ").strip().upper()
    assert dataset_selection in ['ID10M', 'ITU', 'PARSEME'], "Dataset must be one of ID10M, ITU, PARSEME"

    # we dont have test set for ITU dataset
    if mode == "test" and dataset_selection == "ITU":
        assert False, "Test mode is not available for ITU dataset. Please use update mode instead."

    # test ve update için parametrelerin pathini alıcaz,
    # update -> mesela ID10M'de train ettik,  ITU ile parametreleri finetune etcez
    # test -> parametleri freezeleyip test edicez
    # farklı checkpointler yaparız, mesela after ID10M, after ID10M + ITU etc

    if mode in ["test","update"]:
        # load the model
        checkpoint = input("Enter the path of the model: ").strip()
        assert os.path.exists(checkpoint), "Model path does not exist"

    # get stanza tagger for both languages
    tagger_dict = initialize(use_gpu=True)

    # get the path for the dataset
    main_path = r"./resources/"+dataset_selection+"/"
    train_file = main_path + "train.tsv"
    dev_file = main_path + "dev.tsv"
    test_file = main_path + "test.tsv"

    labels_vocab = {"<pad>":0, "B-IDIOM":1, "I-IDIOM":2, "O":3}

    # initialize the dataset
    train_dataset, dev_dataset, test_dataset = None, None, None
    if mode in ["train", "update"]:
        train_dataset = IdiomDataset(train_file, bert_tokenizer, labels_vocab, tagger_dict)
        dev_dataset = IdiomDataset(dev_file, bert_tokenizer, labels_vocab, tagger_dict)
        print(f"train sentences: {len(train_dataset)}")
        print(f"dev sentences: {len(dev_dataset)}")
        print("-" * 50 + "\n")
    else:
        train_dataset = IdiomDataset(train_file, bert_tokenizer, labels_vocab, tagger_dict)
        test_dataset = IdiomDataset(test_file, bert_tokenizer, labels_vocab, tagger_dict) 
        print(f"test sentences: {len(test_dataset)}")
        print("-" * 50 + "\n")


    idioms_train, idioms_dev, idioms_test = None, None, None

    if mode in ["train", "update"]:
        idioms_train = get_idioms(train_dataset, tagger_dict)
        idioms_dev = get_idioms(dev_dataset, tagger_dict)
        print(f"Idioms in train: {len(idioms_train)}")
        print(f"Idioms in dev: {len(idioms_dev)}")
        percentage_elements_in_train_also_in_dev = overlap_percentage_l1_in_l2(idioms_dev, idioms_train)
        print(f"Percentage of idioms in train also in dev: {percentage_elements_in_train_also_in_dev}")
        print("-" * 50 + "\n")
    else:
        idioms_train = get_idioms(train_dataset, tagger_dict)
        idioms_test = get_idioms(test_dataset, tagger_dict)
        print(f"Idioms in test: {len(idioms_test)}")
        percentage_elements_in_train_also_in_test = overlap_percentage_l1_in_l2(idioms_test, idioms_train)
        print(f"Percentage of idioms in train also in test: {percentage_elements_in_train_also_in_test}")
        print("-" * 50 + "\n")

