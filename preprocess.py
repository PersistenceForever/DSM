"""
We need the last function to help extract the final answer of SPARQL, used in check_sparql
此文件最终生成的是output_dir文件中的test.pt,train.pt和val.pt,这三个文件分别存放的是question被tokenizer
生成的id和mask，program对应的id，以及choice和answer在vocab.json中对应的id，即每一个文件包含5部分内容
"""

import os
import json
import pickle
import argparse
import numpy as np
from nltk import word_tokenize
from collections import Counter
from itertools import chain
from tqdm import tqdm
import re
import sys
from transformers import BartTokenizer
from transformers import T5Tokenizer

def get_subkg_seq(subkg): #得到子图的sequence形式，subkg是数据集中的一行数据，即子图
    seq = []
    maskList = []
    g_nodes = subkg['g_node_names']
    g_edges = subkg['g_edge_types']
    g_adj = subkg['g_adj']
    all_subjects = []
    all_objects = []
    for key, value in g_adj.items():
        subject = g_nodes[key]
        all_subjects.append(subject)
        #relation = value.values()
        for k, relation in value.items():
            obj = g_nodes[k]
            all_objects.append(obj)
            #PQ relation is a list 
            relation = relation[0]
            if relation.find('/') >= 0:
                relation = relation.strip().split('/')[-1]
            

            #WQ relation is a str
            # relation = relation.strip().split('/')[-1]

            if relation.find('_')!=-1:
                relation = relation.split('_')
                relation = ' '.join(relation).strip()
            fact = "{} {} {}".format(subject, relation, obj)
            seq.append(fact)
    subkg = ' </s> '.join(seq)#join函数表示用'</s>'作为seq的分割符，如seq=['1', '3']，则返回1 </s> 3
    # maskList = list(set(all_subjects).intersection(set(all_objects)))
    return subkg, seq, maskList  #seq的形式(即subgraph的形式)，不同fact之间通过</s>分割
'''with open('./dataset/test.json','r') as f:    
    for line in f.readlines():
        line = json.loads(line.strip())
        subkg = line['inGraph']
        seq = get_subkg_seq(subkg)
        print(seq)
        sys.exit(0)'''
def encode_dataset(dataset, tokenizer, test=False):
    max_seq_length = 1024
    questions = []
    answers = []
    subkgs = []
    for item in tqdm(dataset):
        question = item['outSeq']
        questions.append(question)
        subkg = item['inGraph']
        subkg, _, _ = get_subkg_seq(subkg)
        subkgs.append(subkg)
        answer = item['answers']       
        if len(answer)>1:
            answer = [', '.join(answer)]
        if len(answer)==0:
            answer = ['']
        answers = answers + answer
    s = [i +' </s> ' + j for i, j in zip(subkgs, answers)] # BART开始的标记是<s>,分割和结尾的token是</s>
    
    input_ids = tokenizer.batch_encode_plus(s, max_length = max_seq_length, pad_to_max_length = True, truncation = True)#与上面的tokenizer()是一样的
    source_ids = np.array(input_ids['input_ids'], dtype = np.int32)
    source_mask = np.array(input_ids['attention_mask'], dtype = np.int32)

    '''with open(os.path.join('./out_test', 'out_mask.txt'), 'a') as f:
        for i in range(len(source_mask)):
            print(len(np.where(source_mask[i]==1)[0]))
            f.write(str(len(np.where(source_mask[i]==1)[0])))
            f.write('\n')'''

    if not test:
        target_ids = tokenizer.batch_encode_plus(questions, max_length = max_seq_length, pad_to_max_length = True, truncation = True)
        target_ids = np.array(target_ids['input_ids'], dtype = np.int32)
    else:
        target_ids = np.array([], dtype = np.int32)
        
    answers = tokenizer(answers, max_length = max_seq_length, pad_to_max_length = True, truncation = True)
    answers = np.array(answers['input_ids'], dtype = np.int32)
    return source_ids, source_mask, target_ids, answers #source_ids,source_mask是subkg+answer通过tokenizer对应的id和mask；target_ids是question对应的id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--model_name_or_path', required=True)
    args = parser.parse_args()
    print('Loading!!!!')
    #test_set,train_set,val_set的类型为list
    test_set = []
    train_set = []
    val_set = []
    with open(os.path.join(args.input_dir, 'test.json')) as f:
        for line in f.readlines():
            line = line.strip()
            line = json.loads(line)
            test_set.append(line)
    with open(os.path.join(args.input_dir, 'train.json')) as f:
        for line in f.readlines():
            line = json.loads(line.strip())
            train_set.append(line)
    with open(os.path.join(args.input_dir, 'dev.json')) as f:
        for line in f.readlines():
            line = json.loads(line.strip())
            val_set.append(line)
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    # BART    
    tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)

    # T5
    # tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    for name, dataset in zip(('train', 'val', 'test'), (train_set, val_set, test_set)):
        print('Encode {} set'.format(name))
        # outputs = encode_dataset(dataset,tokenizer, name == 'test')
        outputs = encode_dataset(dataset,tokenizer, False)
        print(type(outputs))
        print('shape of input_ids of questions, attention_mask of questions, input_ids of sparqls, choices and answers:')
        with open(os.path.join(args.output_dir, '{}.pt'.format(name)), 'wb') as f:
            for o in outputs:
                print(o.shape)
                pickle.dump(o, f)
if __name__ == '__main__':
    main()