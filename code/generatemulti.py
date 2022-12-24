import sys
from ebdataset import EBdataset
from alignment import *
from alignment import align_response,get_doc,write_np
from utils import *
import jsonlines
def tsk1(topcnt):
    tsk1_path='/home/cc/github/Adversarial_Matching/data/entailment_trees_emnlp2021_data_v3/dataset/task_1'
    tsk2_path='/home/cc/github/Adversarial_Matching/data/entailment_trees_emnlp2021_data_v3/dataset/task_2'
    tsk1=EBdataset(tsk1_path,tsk2_path)
    tsk1.get_neg(osp.join(f'data/top{topcnt}','task1.npy'))
    return tsk1

def tsk2(topcnt):
    tsk1_path='/home/cc/github/Adversarial_Matching/data/entailment_trees_emnlp2021_data_v3/dataset/task_1'
    tsk2_path='/home/cc/github/Adversarial_Matching/data/entailment_trees_emnlp2021_data_v3/dataset/task_2'
    tsk1=EBdataset(tsk1_path,tsk2_path)
    tsk1.get_neg(osp.join(f'data/top{topcnt}','task2.npy'))
    return tsk1

def writejson(dir,name,tsk,taskmode,topcnt,tsource):
    assert taskmode in ['golden','distractor']
    mkdir(dir)
    path=osp.join(dir,name)
    with jsonlines.open(path, 'a') as w:
        for data in tsk.data:
            choice=data.neg[:topcnt].tolist()
            source=tsource[data.id,:topcnt].tolist()
            choice.append(data.pos)
            w.write({'id':data.id,'question':data.topara(taskmode),'choice':choice,'source':source})

def main(topcnt):
    tsk1_=tsk1(topcnt)
    tsk2_=tsk2(topcnt)
    source_1=np.load(osp.join(f'data/top{topcnt}','task1source.npy'))
    source_2=np.load(osp.join(f'data/top{topcnt}','task2source.npy'))
    writejson('data/QAdataset','multitask1.jsonl',tsk1_,'golden',topcnt,source_1)
    writejson('data/QAdataset','multitask2.jsonl',tsk2_,'distractor',topcnt,source_2)
    

if __name__=='__main__':
    main(3)

