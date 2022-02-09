# -*- coding: utf-8 -*-
import pickle5 as pickle
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import torch
import os


dic1_path='IND_CD_label1.pickle'

dic2_path='IND_CD_label2.pickle'

with open(dic1_path, 'rb') as f:
    label_dic1=pickle.load(f)
with open(dic2_path, 'rb') as f:
    label_dic2=pickle.load(f)
    label_len=len(label_dic1)
device = torch.device("cpu")
model = BertForSequenceClassification.from_pretrained("beomi/KcELECTRA-base",num_labels=label_len)
model.load_state_dict(torch.load("Model/경총인총_IND_KCELECTRA.pt",map_location="device"),strict=False)
model.to(device)
tokenizer=BertTokenizer.from_pretrained("beomi/KcELECTRA-base")