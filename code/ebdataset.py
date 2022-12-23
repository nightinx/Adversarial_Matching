import numpy as np
from utils import *
import jsonlines
import os
import re
class EBdata:
    def __init__(self,golden=[],distractor=[],pos="",idx=0,label="",neg=[],exclude=set([])):
        self.golden = golden
        self.distractor = distractor
        self.pos = pos
        self.id = idx
        self.label=label
        self.neg=neg
        self.exclude=exclude

    def show(self):
        print(self.tcontext)
        print(self.tlist)
        print(self.pos)
        print(self.id)
        print(self.label)
        print(self.neg)
        print(self.exclude)

    def topara(self,flag):
        assert flag in ['golden','distractor']
        if flag=='golden':
            tlist=self.golden
        else:
            tlist=self.distractor
        para=""
        for i in tlist:
            para+=i.strip()+'. '
        para=para[:-1]
        return para

class EBdataset:
    def __init__(self,tsk1_path,tsk2_path):
        self.data=self.get_data_from_entailment_bank(tsk1_path)
        self.add_distractor(tsk2_path)


    def get_data_from_entailment_bank(self,path):
        datas=[]
        pattern = r'sent\d+: '
        cnt=0
        with jsonlines.open(osp.join(path,'train.jsonl'), "r") as rfd:
            for data in rfd:
                result = re.split(pattern, data['context'])
                result_n = set([x.strip('""').strip() for x in result if x.strip()!=''])
                datas.append(EBdata(list(result_n),[],data['hypothesis'].strip(),cnt,"train"))
                cnt+=1
        rfd.close()     
        with jsonlines.open(osp.join(path,'dev.jsonl'), "r") as rfd:
            for data in rfd:
                result = re.split(pattern, data['context'])
                result_n = set([x.strip('""').strip() for x in result if x.strip()!=''])
                datas.append(EBdata(list(result_n),[],data['hypothesis'].strip(),cnt,"dev"))
                cnt+=1
        rfd.close()  
        with jsonlines.open(osp.join(path,'test.jsonl'), "r") as rfd:
            for data in rfd:
                result = re.split(pattern, data['context'])
                result_n = set([x.strip('""').strip() for x in result if x.strip()!=''])
                datas.append(EBdata(list(result_n),[],data['hypothesis'].strip(),cnt,"test"))
                cnt+=1
                
        rfd.close()
        return datas

    def add_distractor(self,path):
        pattern = r'sent\d+: '
        cnt=0
        with jsonlines.open(osp.join(path,'train.jsonl'), "r") as rfd:
            for data in rfd:
                result = re.split(pattern, data['context'])
                result_n = set([x.strip('""').strip() for x in result if x.strip()!=''])
                self.data[cnt].distractor=list(result_n)
                cnt+=1
        rfd.close()     
        with jsonlines.open(osp.join(path,'dev.jsonl'), "r") as rfd:
            for data in rfd:
                result = re.split(pattern, data['context'])
                result_n = set([x.strip('""').strip() for x in result if x.strip()!=''])
                self.data[cnt].distractor=list(result_n)
                cnt+=1
        rfd.close()  
        with jsonlines.open(osp.join(path,'test.jsonl'), "r") as rfd:
            for data in rfd:
                result = re.split(pattern, data['context'])
                result_n = set([x.strip('""').strip() for x in result if x.strip()!=''])
                self.data[cnt].distractor=list(result_n)
                cnt+=1
                
        rfd.close()

    @staticmethod
    def distractor_include(data1,data2):
        for i in data2.golden:
            if not i in data1.distractor:
                return False
        return True

    @staticmethod
    def golden_include(data1,data2):
        for i in data2.golden:
            if not i in data1.golden:
                return False
        return True

    def get_table(self,func):
        for data1 in self.data:
            data1.exclude=[]
            for data2 in self.data:
                if func(data1,data2):
                    data1.exclude.append(data2.id)
            data1.exclude=set(data1.exclude)

    def get_tsk1_table(self):
        self.get_table(EBdataset.golden_include)

    def get_tsk2_table(self):
        self.get_table(EBdataset.distractor_include)

    def get_neg(self,path):
        a=np.load(path)
        for data in self.data:
            data.neg=a[data.id]


 
            