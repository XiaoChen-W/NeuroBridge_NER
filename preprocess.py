# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 13:27:19 2021

@author: Lenovo
"""

import numpy as np
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics
from copy import deepcopy
from ner_evaluation.ner_eval import collect_named_entities
from ner_evaluation.ner_eval import compute_metrics
from ner_evaluation.ner_eval import compute_precision_recall_wrapper

class token():
    def __init__(self,line):
        line = line.split("\t")
        self.order = line[0]
        self.loc = line[1]
        self.text = line[2]
        self.tag = 'O' if line[-2] == '_' else line[-2].split('#')[-1].split('[')[0]
     
class document():
    def __init__(self,path):
        self.content = open(path,'r',encoding = 'utf-8').read().split("\n")
        self.sens_dict = {}
        for line in self.content:
            if not line.startswith('#') and line:
                t = token(line)
                index = int(t.order.split('-')[0])
                if index not in self.sens_dict:
                    self.sens_dict[index] = []
                self.sens_dict[index].append(t)
        self.sens_list = [ self.sens_dict[key] for key in self.sens_dict.keys()]
    


    def data_formating(self):
        num_sens = len(self.sens_list)
        x,y = [[] for i in range(num_sens)],[[] for i in range(num_sens)]
        for i in range(num_sens):
            for g in range(len(self.sens_list[i])):
                x[i].append(self.sens_list[i][g].text)
                y[i].append(self.sens_list[i][g].tag)
            for g in range(len(self.sens_list[i])):
                if y[i][g] != 'O':
                    if g == 0:
                        y[i][g] = 'B-'+ y[i][g]
                    else:
                        if y[i][g-1].split('-')[-1] != y[i][g]:
                            y[i][g] = 'B-'+ y[i][g]
                        else:
                            y[i][g] = 'I-'+ y[i][g]
        return x,y
    def get_emb(self,emb):
        dic = open(emb,'r',encoding='ISO-8859-1').read()
        dic = dic.split('\n')
        for i in range(len(dic)):
            dic[i] = dic[i].split(' ')
        key, va = [],[]
        for line in dic:
            key.append(line[0])
            va.append(line[1:])
        dic = dict(zip(key,va))
        return dic
    def vectorize(self,x,embedding):
        emb = self.get_emb(embedding)
        vec_x = deepcopy(x)
        for i in range(len(vec_x)):
            for g in range(len(vec_x[i])):
                if x[i][g] in emb.keys():
                    vec_x[i][g] =  list(np.around(np.array(emb[x[i][g]][:-1]).astype("float")*(1),decimals=7).astype("bytes"))  
                else:
                    vec_x[i][g] =  list(np.around(np.array([0 for i in range(50)]).astype("float")*(1),decimals=7).astype("bytes"))
        return vec_x
    def train(self, model = 'CRF'):
        x,y = self.data_formating()
        x = self.vectorize(x, 'D:/research/MedLine_vector_50d_400k.txt')
        if model == 'CRF':
            self.clf = CRF()
            self.clf.fit(x,y)
        return self.clf
            
    def eval(self, test):
        x,y_test = test.data_formating()    
        y_pred = self.clf.predict(x)
        classes = self.clf.classes_    
        classes = list(set([i[2:] for i in classes if i != 'O' and i!= 'nan']))
        metrics_results = {'correct': 0, 'incorrect': 0, 'partial': 0,
                            'missed': 0, 'spurious': 0, 'possible': 0, 'actual': 0, 'precision': 0, 'recall': 0}
        
        # overall results
        results = {'strict': deepcopy(metrics_results),
                    'ent_type': deepcopy(metrics_results),
                    'partial':deepcopy(metrics_results),
                    'exact':deepcopy(metrics_results)
                  }
        
        
        # results aggregated by entity type
        evaluation_agg_entities_type = {e: deepcopy(results) for e in classes}
        for true_ents, pred_ents in zip(y_test, y_pred):
            
            # compute results for one message
            tmp_results, tmp_agg_results = compute_metrics(
                collect_named_entities(true_ents), collect_named_entities(pred_ents),  classes
            )
            
            #print(tmp_results)
        
            # aggregate overall results
            for eval_schema in results.keys():
                for metric in metrics_results.keys():
                    results[eval_schema][metric] += tmp_results[eval_schema][metric]
                    
            # Calculate global precision and recall
                
            results = compute_precision_recall_wrapper(results)
        
        
            # aggregate results by entity type
         
            for e_type in classes:
        
                for eval_schema in tmp_agg_results[e_type]:
        
                    for metric in tmp_agg_results[e_type][eval_schema]:
                        
                        evaluation_agg_entities_type[e_type][eval_schema][metric] += tmp_agg_results[e_type][eval_schema][metric]
                        
                # Calculate precision recall at the individual entity level
                        
                evaluation_agg_entities_type[e_type] = compute_precision_recall_wrapper(evaluation_agg_entities_type[e_type])
        e = evaluation_agg_entities_type
        print('Partial:')
        for keys in e:
            
            p = e[keys]['partial']['precision']
            r = e[keys]['partial']['recall']

            if p == 0 or r == 0:
                f = 0
            else:
                f = round(2*p*r/(p+r),2)
            p,r = round(p,2),round(r,2)
            print(keys,' ',p,' ',r, ' ',f)
        print('Strict:')
        for keys in e:
            
            p = e[keys]['strict']['precision']
            r = e[keys]['strict']['recall']

            if p == 0 or r == 0:
                f = 0
            else:
                f = round(2*p*r/(p+r),2)
            p,r = round(p,2),round(r,2)
            print(keys,' ',p,' ',r, ' ',f)          
            
            
        for col in range(len(y_test)):
            for row in range(len(y_test[col])):
                y_test[col][row] = y_test[col][row].split('-')[-1]
        for col in range(len(y_pred)):  
                for row in range(len(y_pred[col])):
                    y_pred[col][row] = y_pred[col][row].split('-')[-1]
        print(metrics.flat_classification_report(y_test, y_pred,digits=3 ))
    
    
    
if __name__ == '__main__':
        
    train = document("D:/neurobri/Annotation-Project-main/Sample_Annotated_File_Format/Test_2021_09_03/PMC5310843.tsv")
    clf = train.train() 
    test = document("D:/neurobri/Annotation-Project-main/Sample_Annotated_File_Format/Test_2021_09_03/PMC5310843.tsv")
    train.eval(test)
    



