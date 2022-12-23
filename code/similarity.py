from sentence_transformers import SentenceTransformer
import sys
from ebdataset import EBdataset
import numpy as np
import torch
import os
from utils import *
#generate weight matrix of similarity
def similarity(tsk:EBdataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sim_mat=torch.zeros((len(tsk.data),len(tsk.data))).to(device)
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    for data in tsk.data:
        print(data.id)
        pos_sentences = [data.pos]
        neg_sentences=data.neg
        pos_embeddings = model.encode(pos_sentences)
        neg_embeddings= model.encode(neg_sentences)
        pos_embeddings = torch.tensor(pos_embeddings).to(device)
        neg_embeddings=torch.tensor(neg_embeddings).to(device)
        pos_norm=torch.nn.functional.normalize(pos_embeddings, p=2.0, dim = 1)
        neg_norm=torch.nn.functional.normalize(neg_embeddings, p=2.0, dim = 1)
        sim=torch.mm(neg_norm,pos_norm.T)
        sim=sim.view(sim.shape[0])
        sim_mat[data.id]=sim
    
    return sim_mat.detach().cpu().numpy()





