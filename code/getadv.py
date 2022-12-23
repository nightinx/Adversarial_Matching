import sys
from ebdataset import EBdataset
from alignment import *
from alignment import align_response,get_doc,write_np
from utils import *

def tsk1():
    tsk1_path='/home/cc/github/Adversarial_Matching/data/entailment_trees_emnlp2021_data_v3/dataset/task_1'
    tsk2_path='/home/cc/github/Adversarial_Matching/data/entailment_trees_emnlp2021_data_v3/dataset/task_2'
    tsk1=EBdataset(tsk1_path,tsk2_path)
    tsk1.get_neg(osp.join('data/alignment','neg.npy'))
    tsk1.get_tsk1_table()
    return tsk1

def tsk2():
    tsk1_path='/home/cc/github/Adversarial_Matching/data/entailment_trees_emnlp2021_data_v3/dataset/task_1'
    tsk2_path='/home/cc/github/Adversarial_Matching/data/entailment_trees_emnlp2021_data_v3/dataset/task_2'
    tsk1=EBdataset(tsk1_path,tsk2_path)
    tsk1.get_neg(osp.join('data/alignment','neg.npy'))
    tsk1.get_tsk2_table()
    return tsk2

def getadv(tsk:EBdataset,relevance,similarity,lambda_,topcnt,tskmode='task1'):
    assert tskmode in ['task1','task2']
    largenumber=1e10
    weight_matrix=np.log(relevance)+lambda_*np.log(similarity)
    for i in weight_matrix.shape[0]:
        for j in weight_matrix.shape[1]:
            if relevance[i][j]>0.9 or similarity[i][j]>0.85 or j in tsk.data[i].exclude:
                weight_matrix[i][j]=-largenumber

    adv=[]
    for data in tsk.data:
        topnarg=np.argsort(weight_matrix[data.id])
        cnt=0
        negs=set()
        for i in topnarg[::-1]:
            neg=data.neg[i]
            if neg in data.golden or neg in negs or (tskmode=='task2' and neg in data.distractor):
                continue
            negs.add(neg)
            cnt+=1
            if cnt>=topcnt:
                break
        adv.append(list(negs))

    return np.array(adv)
    
def main(topcnt):
    tsk1=tsk1()
    tsk2=tsk2()
    relevance=np.load('data/relevance/test.npy')
    similarity=np.load('data/similarity/test.npy')
    t1np=getadv(tsk1,relevance,similarity,5,topcnt,tskmode='task1')
    t2np=getadv(tsk2,relevance,similarity,5,topcnt,tskmode='task2')
    savenp(f'data/top{topcnt}','task1.npy',t1np)
    savenp(f'data/top{topcnt}','task2.npy',t2np)

if __name__='__main__':
    main(3)

