import torch.nn.functional as F
import numpy as np
import Model 
model=Model.model
model.to("cpu")
tokenizer=Model.tokenizer
label_dic1=Model.label_dic1
label_dic2=Model.label_dic2

def predictor(texts):
    outputs = model(**tokenizer(texts, return_tensors="pt", padding=True))
    probs = F.softmax(outputs.logits,dim=0).detach().numpy()
    idx_list=np.argsort(probs[0])
    return probs[0],idx_list[::-1]