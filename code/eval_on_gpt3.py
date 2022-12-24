from qadataset import *
def main():
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

if __name__=='__main__':
    main()