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
    return tsk1

def getadv(tsk:EBdataset,relevance,similarity,lambda_,topcnt,tskmode='task1'):
    assert tskmode in ['task1','task2']
    largenumber=1e10
    lambda_=np.mean(relevance)/np.mean(similarity)
    weight_matrix=relevance+lambda_*similarity
    for i in range(weight_matrix.shape[0]):
        for j in range(weight_matrix.shape[1]):
            if relevance[i][j]>0.5 or similarity[i][j]>0.85 or similarity[i][j]<0.2 or (j in tsk.data[i].exclude):
                weight_matrix[i][j]=-largenumber

    adv=[]
    advsource=[]
    for data in tsk.data:
        topnarg=np.argsort(weight_matrix[data.id])
        cnt=0
        negs=[]
        source=[]
        for i in topnarg[::-1]:
            neg=data.neg[i]
            if neg in data.golden or neg in negs or (tskmode=='task2' and neg in data.distractor):
                continue
            negs.append(neg)
            source.append(i)
            cnt+=1
            if cnt>=topcnt:
                break
        adv.append(negs)
        advsource.append(source)

    return np.array(adv),np.array(advsource)
    
def main(topcnt):
    relevance=np.load('data/relevance/test.npy')
    similarity=np.load('data/similarity/test.npy')
    tsk_1=tsk1()
    tsk_2=tsk2()
    t1np,source_1=getadv(tsk_1,relevance,similarity,5,topcnt,tskmode='task1')
    t2np,source_2=getadv(tsk_2,relevance,similarity,5,topcnt,tskmode='task2')
    savenp(f'data/top{topcnt}','task1.npy',t1np)
    savenp(f'data/top{topcnt}','task1source.npy',source_1)
    savenp(f'data/top{topcnt}','task2.npy',t2np)
    savenp(f'data/top{topcnt}','task2source.npy',source_2)


if __name__=='__main__':
    main(3)

