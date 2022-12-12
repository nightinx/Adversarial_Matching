import spacy
import os
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")
from utils import unzip

def build_hypo_dictionary(read_path):
    import jsonlines
    import re
    dict_hypo_theoryset={}
    pattern = r'sent\d+: '
    with jsonlines.open(read_path, "r") as rfd:
        for data in rfd:
            result = re.split(pattern, data['context'])
            result_n = set([x.strip() for x in result if x.strip()!=''])
            dict_hypo_theoryset[data['hypothesis']]=result_n
    rfd.close()
    
    dict_hypo_theorypara={}
    for i,hypothesis1 in enumerate(dict_hypo_theoryset.keys()):
        para=""
        for sent in dict_hypo_theoryset[hypothesis1]:
            para+=sent+'. '
        dict_hypo_theorypara[hypothesis1]=para
    return dict_hypo_theoryset,dict_hypo_theorypara

def align_response(hypo_list,theory_list,i,j,nlp,involve_theory=False):
    hypo1=hypo_list[i]
    hypo2=hypo_list[j]
    theory1=theory_list[i]
    theory2=theory_list[j]
    import random
    
    #----------------------------------------------------------------------
    #build word_candidates
    word_candidates=[]
    def delete_all_element(ele):
        for i in range(len(word_candidates)-1,-1,-1): 
            if word_candidates[i] == ele:
                word_candidates.pop(i)
                
    if involve_theory:            
        doc = nlp(hypo1+theory1)
    else:
        doc = nlp(hypo1)
        
    for token in doc:
        # condition to change
        if (not token.pos_ in ['PRON']) and (token.dep_ in ['nsubj','dobj'] or token.pos_ in ['NOUN'] ):
            word_candidates.append(token.text)
    
    #print(word_candidates)
    #----------------------------------------------------------------------        
    #build noun_set
    doc = nlp(theory1)
    noun_set=[]
    for token in doc:
        # condition to change
        if token.pos_ in ['PROPN','NOUN'] or token.dep_ in ['nsubj','dobj']:
            noun_set.append(token.text)      
    noun_set+=word_candidates
    noun_set=set(noun_set)
    
    #----------------------------------------------------------------------
    #substitue hypo
    doc = nlp(hypo2)
    
    attackhypo=""
    
    for token in doc:
        chosen=token.text
        if chosen in word_candidates:
            delete_all_element(chosen)
    #print(word_candidates)            
    for token in doc:
        attackhypo+=" "
        chosen=token.text
        # condition to change
        if token.pos_ in ['PROPN','NOUN'] and token.dep_ in ['nsubj','dobj','nsubjpass','pobj']:
            if token.text not in noun_set and len(word_candidates)>0:
                chosen=random.choice(word_candidates)
                delete_all_element(chosen)
                #print(word_candidates)
        attackhypo+=chosen
    return attackhypo

def show_sent(sent):
    print(sent)
    doc = nlp(sent)
    for token in doc:
        print(token.text,token.pos_,token.dep_)

def write_as_json(path):
    import random
    import jsonlines
    adversay_dict={}
    for i,hypo1 in enumerate(hypo_list):
        adversay_dict[hypo1]=""
        if i>100:
            break
        k_list=random.sample([i for i in range(len(hypo_list))],5)
        for j in k_list:
            if i==j:
                continue
            attackhypo=align_response(hypo_list,theory_list,i,j,nlp)
            adversay_dict[hypo1]+=attackhypo
            adversay_dict[hypo1]+='.'
    with jsonlines.open(path, 'a') as w:
        for hypo,attack_list in adversay_dict.items():
            w.write({hypo: attack_list})

#write_as_json('test.jsonlines')     

def write_as_json_all(hypo_list,theory_list,path):
    import random
    import jsonlines
    adversay_dict={}
    for i,hypo1 in enumerate(hypo_list):
        adversay_dict[hypo1]=""
        for j in range(len(hypo_list)):
            if i==j:
                continue
            attackhypo=align_response(hypo_list,theory_list,i,j,nlp)
            adversay_dict[hypo1]+=attackhypo
            adversay_dict[hypo1]+='.'
        break
    with jsonlines.open(path, 'a') as w:
        for hypo,attack_list in adversay_dict.items():
            w.write({hypo: attack_list})


if __name__=='__main__':
    path='./data'
    if not os.path.exists(path):
        os.mkdir(path)
    #download_entailment_bank(path)
    zip_path="./data/entailment_trees_emnlp2021_data_v3.zip"
    tsk1_path="./data/entailment_trees_emnlp2021_data_v3/dataset/task_1"
    train_path=os.path.join(tsk1_path,'train.jsonl')
    unzip(zip_path,path)
    dict_set,dict_para=build_hypo_dictionary(train_path)
    hypo_list=[]
    theory_list=[]
    for i,hypothesis1 in enumerate(dict_para.keys()):
        hypo_list.append(hypothesis1)
        theory_list.append(dict_para[hypothesis1])
    write_as_json_all(hypo_list,theory_list,'./data/aligened_tree.jsonlines')
