# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 16:49:23 2020

@author: truthless
"""
from LAUG.nlu.gpt.utils import seq2dict
from LAUG.nlu.milu_new.dai_f1_measure import DialogActItemF1Measure


def normalize(data):
    string = str(data)
    
    digit2word = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five',
        '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten', '11': 'eleven',
        '12': 'twelve'
    }

    for key, value in digit2word.items():
        string = string.replace(' ' + key + ' ', ' ' + value + ' ')
    return eval(string)


def calculateF1gpt(data):
    data = normalize(data)
    dai_f1_metric = DialogActItemF1Measure()
    
    for item in data:
        predict = seq2dict(item[1].replace('\'','').replace('=?','= ?').lower())
        target = seq2dict(item[0].replace(' \'','').split('&')[1])
        dai_f1_metric([predict], [target])
    
    metric = dai_f1_metric.get_metric(True)
    print(metric)
def calculateF1copy(data):
    data = normalize(data)
    dai_f1_metric = DialogActItemF1Measure()
    
    for item in data:
        predict = seq2dict(item[2].replace('i d','id').lower())
        target = seq2dict(item[1])
        dai_f1_metric([predict], [target])
    
    metric = dai_f1_metric.get_metric(True)
    print(metric)

if __name__ == '__main__':
    from sys import argv
    import json

    # files are JSON outputs from run.py file 
    # if usage is: python evaluate_g.py file1 file2 
    if len(argv) >3: 
        if argv[3]=='gpt':
            diffs =[]
            with open(argv[1], 'r', encoding='utf-8') as f:
                data_orig=json.load(f)
            data_orig = normalize(data_orig)
    
            with open(argv[2], 'r', encoding='utf-8') as f:
                data_aug=json.load(f)
            data_aug = normalize(data_aug)
            val_pred_count=0
            for item1, item2 in zip(data_orig, data_aug):
                # if any of the lines have an invalid prediction, skip them 
                try: 
                    predict1 = seq2dict(item1[1].replace('=?','= ?').lower())
                    predict2 = seq2dict(item2[1].replace('=?','= ?').lower())
                    target1 = item1[0].replace(' \'','').split('&')[1]
                    target2 = item2[0].replace(' \'','').split('&')[1]
                    input1 = item1[0].replace(' \'','').split('&')[0]
                    input2 = item2[0].replace(' \'','').split('&')[0]

                    assert target1 == target2, f"Target output is different: {(target1, target2)}"

                    target = seq2dict(target1)

                    # keep track of only those where the prediction on the paraphrased test set is wrong and those for the original test set is correct 
                    if predict1 != predict2 and predict2 != target and predict1 == target: 
                        # print("add sample")
                        diffs.append({
                            "original_input": input1,
                            "paraphrased_input": input2, 
                            "original_prediction": item1[1].replace('=?','= ?').lower(), 
                            "paraphrased_prediction": item2[1].replace('=?','= ?').lower(), 
                            "target": target1
                        })
                    val_pred_count += 1 
                except: 
                    continue
            with open("diffs.json", "w") as f: 
                json.dump(diffs, f, indent=4)
            print(f"Number of valid predictions from both files: {val_pred_count}\n\tTotal number of predictions: {len(diffs)}\n\tNumber of invalid predictions from either file: {len(diffs)- val_pred_count}")

    # if only one file is provided: i.e. python evaluate_g.py <file>.json, return the F1 score 
    else: 
        data=[]
        if argv[2]=='gpt':
            with open(argv[1], 'r', encoding='utf-8') as f:
                data=json.load(f)
            calculateF1gpt(data)
        if argv[2]=='copy':
            with open(argv[1], 'r', encoding='utf-8') as f:
                data=json.load(f)
            calculateF1copy(data)