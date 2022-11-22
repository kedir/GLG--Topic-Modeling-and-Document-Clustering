import pandas as pd
import numpy as np
import re
import os
import pickle
import torch
import transformers
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig
from tqdm import tqdm, trange
from keras_preprocessing.sequence import pad_sequences
from transformers import BertForTokenClassification, AdamW

class nerModel:
    
  def __init__(self, model_path):
    self.ner_model = {}
    self.idx2tag = pickle.load(open(os.path.join(model_path, "idx2tag.pkl"), 'rb'))
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)
    if torch.cuda.is_available():
        self.model = torch.load(os.path.join(model_path,"model.pt"))
    else:
        self.model = torch.load(os.path.join(model_path,"model.pt"), map_location=torch.device('cpu'))
    self.model.eval()
    
  def do_pridict(self, input_sentence):
    result = {}
    # first toknize the sentences
    tokenized_sentence = self.tokenizer.encode(input_sentence)
    if torch.cuda.is_available():
        input_ids = torch.tensor([tokenized_sentence]).cuda()
    else:
        input_ids = torch.tensor([tokenized_sentence])
    # run the sentences through the model
    with torch.no_grad():
        output = self.model(input_ids)
    label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)

    # join bpe split tokens
    tokens = self.tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
    new_tokens, new_labels = [], []
    for token, label_idx in zip(tokens, label_indices[0]):
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_labels.append(self.idx2tag[label_idx])
            # print(label_idx)
            new_tokens.append(token)
    result['tokens'] = new_tokens
    result['labels'] = new_labels
    return result
      
