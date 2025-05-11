import os
from tqdm import tqdm
import pandas as pd
import pickle
import torch
from torch import nn
from torch.utils.data import Dataset
from pprint import pprint
from collections import Counter
import random
import numpy as np
from typing import List, Dict
import json
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaConfig
#from TorchCRF import CRF
from torchcrf import CRF
import unicodedata
import re
import torch.nn.functional as F
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import io
import seaborn as sns
