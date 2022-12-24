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

def main():
    tsk1_path='/home/cc/github/Adversarial_Matching/data/entailment_trees_emnlp2021_data_v3/dataset/task_1'
    tsk2_path='/home/cc/github/Adversarial_Matching/data/entailment_trees_emnlp2021_data_v3/dataset/task_2'
    tsk1=EBdataset(tsk1_path,tsk2_path)
    tsk1.get_neg(osp.join('data/alignment','neg.npy'))
    tsk1.get_tsk1_table()
    a=similarity(tsk1)
    savenp('data/similarity','test.npy',a)

if __name__=='__main__':
    main()





