from qadataset import *
import os
import openai
import time
import random
from ebdataset import *
from api_key import get_api_key
def test():
    path='data/QAdataset/multitask1.jsonl'
    QA=QAdataset(path)
    relevance=np.load('data/relevance/test.npy')
    similarity=np.load('data/similarity/test.npy')
    for qa in QA.data[:5]:
        print(qa.question)
        print(qa.choice)
        print(qa.id)
        print(qa.source)
        for j in qa.source:
            print(relevance[qa.id][j])
            print(similarity[qa.id][j])

def prompt(data):
    para=""
    para+="Based on the statements that:"
    para+=f"\n{data.question}"
    para+=f"\n\nWhich of the following conclusions can be inferred?\n\n"
    for index,choice in enumerate(data.choice):
        para+=f"{index}.{choice}.\n"
    para+="\nA:"
    return para

def shot_prompt(data):
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
    para+="\nA:"
    para+=f'{answer}'
    return para

def final_prompt(data,fewshots):
    para=""
    for shot in fewshots:
        para+=shot_prompt(shot)+'\n\n'
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
            max_tokens=1,
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
    tsk1_path='/home/cc/github/Adversarial_Matching/data/entailment_trees_emnlp2021_data_v3/dataset/task_1'
    tsk2_path='/home/cc/github/Adversarial_Matching/data/entailment_trees_emnlp2021_data_v3/dataset/task_2'
    tsk1=EBdataset(tsk1_path,tsk2_path)
    tsk2=EBdataset(tsk1_path,tsk2_path)
    tsk1.get_tsk1_table()
    tsk2.get_tsk1_table()


    openai.api_key = get_api_key()

    path='data/QAdataset/multitask1.jsonl'
    QA=QAdataset(path)
    with open('data/evaluate/tsk1_fewshots.csv','a') as f:
        f.write(f"id,text,shot_index\n")
        for data in QA.data:
            sampled=sample_shots(data.id,5,tsk1)
            few_shots=[QA.data[i] for i in sampled]
            text=getresponse(final_prompt(data,few_shots))
            #text='3'
            shot_index=str(sampled).replace(',',';').strip('[').strip(']')
            print(f"{data.id},{text},{shot_index}")
            f.write(f"{data.id},{text},{shot_index}\n")
    f.close()

    path='data/QAdataset/multitask2.jsonl'
    QA=QAdataset(path)
    with open('data/evaluate/tsk2_fewshots.csv','a') as f:
        f.write(f"id,text,shot_index\n")
        for data in QA.data:
            sampled=sample_shots(data.id,5,tsk2)
            few_shots=[QA.data[i] for i in sampled]
            text=getresponse(final_prompt(data,few_shots))
            #text='3'
            shot_index=str(sampled).replace(',',';').strip('[').strip(']')
            print(f"{data.id},{text},{shot_index}")
            f.write(f"{data.id},{text},{shot_index}\n")
    f.close()
           
    

if __name__=='__main__':
    main()