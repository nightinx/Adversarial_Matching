from qadataset import *
import os
import openai
import time
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
    return text,response

def main():
    path='data/QAdataset/multitask1.jsonl'
    QA=QAdataset(path)
    openai.api_key = os.getenv("OPENAI_API_KEY")

    start_sequence = "\nA:"
    restart_sequence = "\n\nQ: "
    
    for i in range(100):
        text,_=getresponse(prompt(QA.data[i]))
        print(text)
        time.sleep(10)
    

if __name__=='__main__':
    main()