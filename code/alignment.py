import spacy
from utils import *
from ebdataset import *
import random
import numpy as np
def delete_all_element(word_candidates,ele):
        for i in range(len(word_candidates)-1,-1,-1): 
            if word_candidates[i] == ele:
                word_candidates.pop(i)

class Doc:
    def __init__(self,nlp,data):
        self.pos=nlp(data.pos)
        self.golden=nlp(data.topara('golden'))
        self.id=data.id

def get_doc(tsk,nlp):
    doc=[]
    for data in tsk:
        # print(data.id)
        doc.append(Doc(nlp,data))
    return doc


def align_response(data1,data2,tdoc):
    hypo1=data1.pos
    hypo2=data2.pos
    theory1=data1.topara('golden')
    theory2=data2.topara('golden')
    
    #----------------------------------------------------------------------
    #build word_candidates
    word_candidates=[]
                
    doc = tdoc[data1.id].pos
        
    for token in doc:
        # condition to change
        if (not token.pos_ in ['PRON']) and (token.dep_ in ['nsubj','dobj'] or token.pos_ in ['NOUN'] ):
            word_candidates.append(token.text)
    
    #print(word_candidates)
    #----------------------------------------------------------------------        
    #build noun_set
    doc = tdoc[data1.id].golden
    noun_set=[]
    for token in doc:
        # condition to change
        if token.pos_ in ['PROPN','NOUN'] or token.dep_ in ['nsubj','dobj']:
            noun_set.append(token.text)      
    noun_set+=word_candidates
    noun_set=set(noun_set)
    
    #----------------------------------------------------------------------
    #substitue hypo
    doc = tdoc[data2.id].pos
    
    attackhypo=""
    
    for token in doc:
        chosen=token.text
        if chosen in word_candidates:
            delete_all_element(word_candidates,chosen)
    #print(word_candidates)            
    for token in doc:
        attackhypo+=" "
        chosen=token.text
        # condition to change
        if token.pos_ in ['PROPN','NOUN'] and token.dep_ in ['nsubj','dobj','nsubjpass','pobj']:
            if token.text not in noun_set and len(word_candidates)>0:
                chosen=random.choice(word_candidates)
                delete_all_element(word_candidates,chosen)
                #print(word_candidates)
        attackhypo+=chosen
    return attackhypo.strip()

def write_np(tsk,pdir,name):
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_sm")
    tdoc=get_doc(tsk,nlp)
    print('nlp ends---------------------------------------------------------------')
    mkdir(pdir)
    for data1 in tsk:
        print(data1.id)
        data1.neg=[]
        for data2 in tsk:
            attackhypo=align_response(data1,data2,tdoc)
            data1.neg.append(attackhypo)
    np.save(osp.join(pdir,name),np.array([data.neg for data in tsk]))

def main():
    tsk1_path='/home/cc/github/Adversarial_Matching/data/entailment_trees_emnlp2021_data_v3/dataset/task_1'
    tsk2_path='/home/cc/github/Adversarial_Matching/data/entailment_trees_emnlp2021_data_v3/dataset/task_2'
    tsk1=EBdataset(tsk1_path,tsk2_path)
    write_np(tsk1.data,'data/align_v2','neg.npy')

if __name__=='__main__':
    main()