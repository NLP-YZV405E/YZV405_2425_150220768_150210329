from __init__ import *
from dataset import IdiomDataset
from collate import collate
from model import IdiomExtractor
from bert_embedder import BERTEmbedder
from hparams import HParams
from trainer import Trainer
from utils import *
if __name__ == "__main__":
    
    SEED = 2
    # set seeds to get reproducible results
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    # gpuda bazen randomluk olabiliyormuş onu kaldırmak için
    torch.backends.cudnn.deterministic = True
    CUDA_LAUNCH_BLOCKING=1

    #instantiate the hyperparameters
    params = HParams()

    # # create bert
    # it_model_name = 'dbmdz/bert-base-italian-cased'
    # # output hidden states -> it helps to get hidden states from bert
    # it_config = BertConfig.from_pretrained(it_model_name, output_hidden_states=True)
    # it_tokenizer = BertTokenizer.from_pretrained(it_model_name)
    # # get bert weights
    # hf_it_model = BertModel.from_pretrained(it_model_name, config=it_config)


    # Türkçe BERT
    tr_model_name = "dbmdz/bert-base-turkish-128k-cased"
    tr_config = BertConfig.from_pretrained(tr_model_name, output_hidden_states=True)
    tr_tokenizer = BertTokenizer.from_pretrained(tr_model_name)
    hf_tr_model = BertModel.from_pretrained(tr_model_name, config=tr_config)

    # create bert
    it_model_name = 'dbmdz/bert-base-italian-xxl-cased'
    it_config = AutoConfig.from_pretrained(it_model_name, output_hidden_states=True)
    it_tokenizer = AutoTokenizer.from_pretrained(it_model_name)
    hf_it_model = AutoModel.from_pretrained(it_model_name, config=it_config)

    # # Turkish E5-Large
    # tr_model_name = "ytu-ce-cosmos/turkish-e5-large"
    # tr_config = AutoConfig.from_pretrained(tr_model_name, output_hidden_states=True)
    # tr_tokenizer = AutoTokenizer.from_pretrained(tr_model_name)
    # hf_tr_model = AutoModel.from_pretrained(tr_model_name, config=tr_config)

    # train, update or test mode selection
    mode = input("Do you want to train or test the model? (train, update, test): ").strip().lower()
    assert mode in ['train', 'update', 'test'], "Mode must be one of train, update, test"
    # select the dataset
    dataset_selection = input("Select the dataset (ID10M, ITU, PARSEME, COMBINED): ").strip().upper()
    assert dataset_selection in ['ID10M', 'ITU', 'PARSEME', 'COMBINED'], "Dataset must be one of ID10M, ITU, PARSEME, COMBINED"

    # check dataset path
    tr_path = r"./src/checkpoints/tr/"
    it_path = r"./src/checkpoints/it/"
    os.makedirs(tr_path, exist_ok=True)
    os.makedirs(it_path, exist_ok=True)

    if mode in ["test","update"]:
        # list available checkpoints
        print("Available tr checkpoints:")
        checkpoints_tr = os.listdir(tr_path)
        for i, checkpoint_tr in enumerate(checkpoints_tr):
            print(f"{i+1}. {checkpoint_tr}")
        print("none")
        # load the model
        checkpoint_tr = input("Enter the checkpoint (without .pt): ").strip()
        if checkpoint_tr == "none":
            tr_path = None
        else:
            tr_path = tr_path + checkpoint_tr + ".pt"
            assert os.path.exists(tr_path), "Model path does not exist"

        print("\n")

        print("Available it checkpoints:")
        checkpoints_it = os.listdir(it_path)
        for i, checkpoint_it in enumerate(checkpoints_it):
            print(f"{i+1}. {checkpoint_it}")
        print("none")
        # load the model
        checkpoint_it = input("Enter the checkpoint (without .pt): ").strip()
        if checkpoint_it == "none":
            it_path = None
        else:
            it_path = it_path + checkpoint_it + ".pt"
            assert os.path.exists(it_path), "Model path does not exist"

    model_name = None
    if mode in ["train", "update"]:
        model_name = input("Enter the model name (without .pt): ").strip()

    elif mode == "test":
        model_name = checkpoint_it

    # get stanza tagger for both languages
    tagger_dict = initialize(use_gpu=True)

    # get the path for the dataset
    main_path = r"./resources/"+dataset_selection+"/"
    train_file = main_path + "train.tsv"
    dev_file = main_path + "dev.tsv"
    test_file = main_path + "test.tsv"


    if params.num_classes == 3:
        print("3 classes")
        labels_vocab = {"<pad>":0, "B-IDIOM":1, "I-IDIOM":1, "O":2}

    elif params.num_classes == 4:
        labels_vocab = {"<pad>":0, "B-IDIOM":1, "I-IDIOM":2, "O":3}

    # initialize the dataset
    train_dataset, dev_dataset, test_dataset = None, None, None
    if mode in ["train", "update"]:
        train_dataset = IdiomDataset(train_file, labels_vocab, tagger_dict)
        dev_dataset = IdiomDataset(dev_file, labels_vocab, tagger_dict)
        test_dataset = IdiomDataset(test_file, labels_vocab, tagger_dict, is_test=True) 
        print(f"train sentences: {len(train_dataset)}")
        print(f"dev sentences: {len(dev_dataset)}")
        print("-" * 50 + "\n")
    else:
        test_dataset = IdiomDataset(test_file, labels_vocab, tagger_dict, is_test=True) 
        print(f"test sentences: {len(test_dataset)}")
        print("-" * 50 + "\n")

    

    if dataset_selection == "COMBINED":
        batch_size = 128
    else:
        batch_size = params.batch_size

    #dataloader

    if mode in ["train", "update"]:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
        dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=collate)
        test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=collate)
        print(f"length of train dataloader: {len(train_dataloader)}")
        print(f"length of dev dataloader: {len(dev_dataloader)}")
    else:
        test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=collate)
        print(f"length of test dataloader: {len(test_dataloader)}")


    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    it_embedder =  BERTEmbedder(hf_it_model, it_tokenizer, DEVICE)
    tr_embedder =  BERTEmbedder(hf_tr_model, tr_tokenizer, DEVICE)

    #instantiate the model
    it_model = IdiomExtractor(hf_it_model, it_embedder, params).cuda()

    # it_model.freeze_bert()
    it_model.unfreeze_bert()


    tr_model = IdiomExtractor(hf_tr_model, tr_embedder, params).cuda()

    # tr_model.freeze_bert()
    tr_model.unfreeze_bert()

    if mode in ["update", "test"]: 

        if it_path is not None:
            it_state = torch.load(it_path, map_location=DEVICE)

            # now load cleanly
            it_model.load_state_dict(it_state)

        if tr_path is not None:
            tr_state = torch.load(tr_path, map_location=DEVICE)
            tr_model.load_state_dict(tr_state)

    tr_optimizer = optim.AdamW(
        list(tr_model.parameters()),
        lr=params.lr,
        weight_decay=params.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    it_optimizer = optim.AdamW(
        list(it_model.parameters()),
        lr=params.lr,
        weight_decay=params.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    trainer = Trainer(tr_model = tr_model, it_model = it_model,
                tr_optimizer = tr_optimizer,
                it_optimizer = it_optimizer,
                modelname = model_name,
                labels_vocab=labels_vocab)

    if mode in ["train", "update"]:
        trainer.train(train_dataloader, dev_dataloader, params.epoch, patience=15)
        trainer.test(test_dataloader)
        
    if mode == "test":
        trainer.test(test_dataloader)
    