from sentence_transformers import SentenceTransformer
import sys
sys.path.append("./bert_finetuning/code")
from dataset import get_data
import numpy as np
import torch
import csv
import os

#generate negative hypothesis
def compute_weight(relevance,similarty,lambda_):
    return relevance+lambda_*(1-similarty)


def save_topn(topn,n,path):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+f'/top{n}.csv','w') as f:
        header = ['pos']+[f"neg{i}" for i in range(n)]
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(topn)

def get_adv(n,neg_length,weight_mat,data):
    topn=[]
    pos=data['pos']
    neg=np.array(data['neg'])
    for i in range(neg_length):
        topnarg=np.argpartition(weight_mat[i], -n)[-n:]
        neg_sents=[item.replace(',',';') for item in neg[i][topnarg]]
        topn.append([pos[i]]+neg_sents)
    
    save_topn(topn,n,'./data/topn')

largenumber=1e10
if __name__=='__main__':
    read_path1='./data/entailment_trees_emnlp2021_data_v3/dataset/task_1/train.jsonl'
    read_path2='./data/aligened_tree/aligened_tree.jsonlines'
    data=get_data(read_path1,read_path2)
    relevance=np.load('./data/relevance/relevance.npy')
    print(type(relevance))
    similarity=np.load('./data/similarity/similarity.npy')
    neg_length=len(data['neg'][0])
    weight_mat=compute_weight(relevance,similarity,-0.2)
    for i in range(1276):
        weight_mat[i][-1]=-largenumber
    for i in range(1276):
        for j in range(1276):
            if relevance[i][j]>0.95:
                weight_mat[i][j]=-largenumber
    for i in range(3,10):
        get_adv(i,neg_length,weight_mat,data)





    


