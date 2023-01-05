from qadataset import *
import os
import openai
import time
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

def main():
    openai.api_key = get_api_key()

    path='data/QAdataset/multitask1.jsonl'
    QA=QAdataset(path)
    with open('data/evaluate/tsk1.csv','a') as f:
        f.write(f"id,text\n")
        for data in QA.data:
            text=getresponse(prompt(data))
            print(f"{data.id},{text}")
            f.write(f"{data.id},{text}\n")
    f.close()

    path='data/QAdataset/multitask2.jsonl'
    QA=QAdataset(path)
    with open('data/evaluate/tsk2.csv','a') as f:
        f.write(f"id,text\n")
        for data in QA.data:
            text=getresponse(prompt(data))
            print(f"{data.id},{text}")
            f.write(f"{data.id},{text}\n")
    f.close()
           
    

if __name__=='__main__':
    main()