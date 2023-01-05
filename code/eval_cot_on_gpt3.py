from qadataset import *
import os
import openai
import time
import random
from ebdataset import *
import re
import sys
from api_key import get_api_key
def load_shots(path):
    shots=[]
    with open(path,'r') as f:
        for line in f.readlines()[1:]:
            shotlist=line.strip().split(',')[-1].split(';')
            shot=[int(i) for i in shotlist]
            shots.append(shot)
    return shots

def get_cot(path):
    cots=[]
    pattern = r'sent\d+: '
    cnt=0
    with jsonlines.open(osp.join(path,'train.jsonl'), "r") as rfd:
        for data in rfd:
            result = re.split(pattern, data['context'])
            result_n = [x.strip('""').strip() for x in result if x.strip()!='']
            proof=data['proof']
            for i in range(1,len(result_n)+1):
                proof=proof.replace(f"sent{i}", result_n[i-1])
            proof=proof.replace(f"-> hypothesis; ", f"-> {data['hypothesis']}.")
            cots.append(proof)
            cnt+=1
    rfd.close()     
    with jsonlines.open(osp.join(path,'dev.jsonl'), "r") as rfd:
        for data in rfd:
            result = re.split(pattern, data['context'])
            result_n = [x.strip('""').strip() for x in result if x.strip()!='']
            proof=data['proof']
            for i in range(1,len(result_n)+1):
                proof=proof.replace(f"sent{i}", result_n[i-1])
            proof=proof.replace(f"-> hypothesis; ", f"-> {data['hypothesis']}.")
            cots.append(proof)
            cnt+=1
    rfd.close()   
    with jsonlines.open(osp.join(path,'test.jsonl'), "r") as rfd:
        for data in rfd:
            result = re.split(pattern, data['context'])
            result_n = [x.strip('""').strip() for x in result if x.strip()!='']
            proof=data['proof']
            for i in range(1,len(result_n)+1):
                proof=proof.replace(f"sent{i}", result_n[i-1])
            proof=proof.replace(f"-> hypothesis; ", f"-> {data['hypothesis']}.")
            cots.append(proof)
            cnt+=1
    rfd.close()   
    return cots

def load_entailment_tree(path):
    shots=[]
    with open(path,'r') as f:
        for line in f.readlines()[1:]:
            shotlist=line.strip().split(',')[-1].split(';')
            shot=[int(i) for i in shotlist]
            shots.append(shot)
    return shots

def prompt(data):
    para=""
    para+="Based on the statements that:"
    para+=f"\n{data.question}"
    para+=f"\n\nWhich of the following conclusions can be inferred?\n\n"
    for index,choice in enumerate(data.choice):
        para+=f"{index}.{choice}.\n"
    para+="\nA:"
    return para

def shot_prompt(cots,data):
    para=""
    para+="Q:\nBased on the statements that:"
    para+=f"\n{data.question}"
    para+=f"\n\nWhich of the following conclusions can be inferred?\n\n"
    choices=[(index,choice) for index,choice in enumerate(data.choice)]
    random.shuffle(choices)
    answer=-1
    for index,choice in enumerate(choices):
        para+=f"{index}.{choice[1]}.\n"
        if choice[0]==3:
            answer=index
    para+=f"\nA:{answer}. Reason:"
    para+=f'{cots[data.id]}'
    return para

def final_prompt(data,fewshots,cots):
    para=""
    for shot in fewshots:
        para+=shot_prompt(cots,shot)+'\n\n'
    para+='Q:\n'+prompt(data)

    return para


def getresponse(prompt):
    i=0
    while i<4:
        time.sleep(i*10)
        try:
            response=openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0,
            max_tokens=300,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n"], 
            )
            text=response["choices"][0]["text"]
            return text
        except:
            i+=1
    return "error"

def sample_shots(id,cnt,tsk):
    total=0
    sampled=set()
    while total<cnt:
        sample=random.sample(range(0,len(tsk.data)),1)[0]
        if not sample in tsk.data[id].exclude and not id in tsk.data[sample].exclude and not sample in sampled:
            total+=1
            sampled.add(sample)
    return list(sampled)

def main():
    tsk1_path='./data/entailment_trees_emnlp2021_data_v3/dataset/task_1'
    tsk2_path='./data/entailment_trees_emnlp2021_data_v3/dataset/task_2'
    tsk1_fewshots_path='./data/evaluate/tsk1_fewshots.csv'
    tsk2_fewshots_path='./data/evaluate/tsk2_fewshots.csv'
    tsk1=EBdataset(tsk1_path,tsk2_path)
    tsk2=EBdataset(tsk1_path,tsk2_path)
    tsk1.get_tsk1_table()
    tsk2.get_tsk1_table()

    cots=get_cot(tsk1_path)
    tsk1_shots=load_shots(tsk1_fewshots_path)
    tsk2_shots=load_shots(tsk2_fewshots_path)

    openai.api_key = get_api_key()

    path='./data/QAdataset/multitask1.jsonl'
    QA=QAdataset(path)
    with open('data/evaluate/tsk1_cot.csv','a') as f:
        f.write(f"id,text,shot_index\n")
        for data in QA.data:
            sampled=tsk1_shots[data.id]
            few_shots=[QA.data[i] for i in sampled]
            text=getresponse(final_prompt(data,few_shots,cots))
            shot_index=str(sampled).replace(',',';').strip('[').strip(']')
            print(f"{data.id},{text[0]},{shot_index}")
            text.replace(',',';').strip()
            f.write(f"{data.id},{text[0]},{shot_index},{text}\n")
    f.close()

    path='./data/QAdataset/multitask2.jsonl'
    QA=QAdataset(path)
    with open('data/evaluate/tsk2_cot.csv','a') as f:
        f.write(f"id,text,shot_index\n")
        for data in QA.data:
            sampled=tsk2_shots[data.id]
            few_shots=[QA.data[i] for i in sampled]
            text=getresponse(final_prompt(data,few_shots,cots))
            shot_index=str(sampled).replace(',',';').strip('[').strip(']')
            print(f"{data.id},{text[0]},{shot_index}")
            text.replace(',',';').strip()
            f.write(f"{data.id},{text[0]},{shot_index},{text}\n")
    f.close()
           
    

if __name__=='__main__':
    main()