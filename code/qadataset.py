import numpy as np
from utils import *
import jsonlines
import os
import re
class QAdata:
    def __init__(self,theory,choice=[],source=[],idx=0):
        self.question = theory
        self.choice = choice
        self.source = source
        self.id=idx


    def show(self):
        print(self.question)
        print(self.choice)
        print(self.id)
        print(self.source)

class QAdataset:
    def __init__(self,path):
        self.data=self.get_qa(path)
        self.choice_cnt=len(self.data[0].choice)


    def get_qa(self,path):
        datas=[]
        cnt=0
        with jsonlines.open(path, "r") as rfd:
            for data in rfd:
                datas.append(QAdata(data['question'],data['choice'],data['source'],data['id']))
                cnt+=1
        rfd.close()     
        return datas

