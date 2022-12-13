from sentence_transformers import SentenceTransformer
import sys
sys.path.append("./bert_finetuning/code")
from dataset import get_data
import numpy as np
import torch
import os

#generate weight matrix of similarity

read_path1='./data/entailment_trees_emnlp2021_data_v3/dataset/task_1/train.jsonl'
read_path2='./data/aligened_tree/aligened_tree.jsonlines'
data=get_data(read_path1,read_path2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
neg_len=len(data['neg'][0])
# neg_len=36
sim_mat=torch.zeros((neg_len,neg_len)).to(device)
model = SentenceTransformer('hiiamsid/sentence_similarity_hindi')
for i in range(neg_len):
    print(i)
    pos_sentences = [data['pos'][i]]
    neg_sentences=data['neg'][i][:neg_len]
    pos_embeddings = model.encode(pos_sentences)
    neg_embeddings= model.encode(neg_sentences)
    pos_embeddings = torch.tensor(pos_embeddings).to(device)
    neg_embeddings=torch.tensor(neg_embeddings).to(device)
    pos_norm=torch.nn.functional.normalize(pos_embeddings, p=2.0, dim = 1)
    neg_norm=torch.nn.functional.normalize(neg_embeddings, p=2.0, dim = 1)
    # print(pos_norm.shape)
    # print(neg_norm.shape)
    sim=torch.mm(neg_norm,pos_norm.T)
    sim=sim.view(sim.shape[0])
    sim_mat[i]=sim
sim_mat=sim_mat.detach().cpu().numpy()
save_dir='./data/similarity'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path=os.path.join(save_dir,'similarity')
np.save(save_path, sim_mat, allow_pickle=True, fix_imports=True)
