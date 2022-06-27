import numpy as np
import json
import pickle
import torch
import os
from random import choice
from bert_embedding import BertEmbedding
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import sys
import random
# import matplotlib.pyplot as plt
import  argparse
np.random.seed(222)
# calculate graph edit distance
def relevance_ged(mode, input_dir, k_shot):
    trainPath = os.path.join(input_dir, 'train.json')
    kbList = [] # all sub-kb's set, every element is a sub_kb
    all_index = [] #all fact's the most related k_shot index  
    with open(trainPath) as f1:
        for line in f1.readlines():
            line = json.loads(line)
            g_adj = line['inGraph']['g_adj'] #g_adj is a dict list
            temp_kb_edge = []
            for k, v in g_adj.items():
                v_node = list(v.keys())
                for vv in v_node:
                    tx = []
                    tx.append(k)
                    tx.append(vv)
                    temp_kb_edge.append(tuple(tx))
            G = nx.DiGraph()
            G.add_edges_from(temp_kb_edge)
            kbList.append(G) #sub_kb is completed
    length = len(kbList)
    print("length:", length)
    train_Score = np.array(length, length)
    if mode=='train':
        for index in range(length):
            
            query_kb = kbList[index]
                       

            for i, reList in enumerate(relationList):
                if i!=index:                    
                    edit_distance = nx.graph_edit_distance(query_kb, kbList[i])
                    scoreDict[i] = edit_distance
            k_shot_Index = [] # top k_shot revelance index in raw training data    
            scoreList = sorted(scoreDict.items(),key = lambda x:x[1],reverse = True)
            scoreList = scoreList[:k_shot]
            for ind, _ in scoreList:
                k_shot_Index.append(ind)
            all_index.append(k_shot_Index)
    if mode =='test':
        testPath = os.path.join(input_dir, mode+'.json')
        test_relationList = []
        test_kbList = []
        with open(testPath) as f2:
            for line in f2.readlines():
                res = set()
                embList = []
                line = json.loads(line)
                relation = list(line['inGraph']['g_edge_types'].values())
                g_adj = line['inGraph']['g_adj'] #g_adj is a dict list
                temp_kb_edge = []
                for k, v in g_adj.items():
                    v_node = list(v.keys())
                    for vv in v_node:
                        tx = []
                        tx.append(k)
                        tx.append(vv)
                        temp_kb_edge.append(tuple(tx))
                G = nx.DiGraph()
                G.add_edges_from(temp_kb_edge)
                test_kbList.append(G) #sub_kb is completed
               
                for _, emb in result:
                    emb = np.array(emb)
                    emb = np.sum(emb, axis=0)
                    embList.append(emb)
                test_relationList.append(embList)
        test_len = len(test_relationList)
        for index in range(test_len):
            query_re = np.array(test_relationList[index])
            query_kb = test_kbList[index]
            scoreDict = dict() 
            for i, reList in enumerate(relationList):         
                edit_distance = nx.graph_edit_distance(query_kb, kbList[i])
                scoreDict[i] = edit_distance
            k_shot_Index = []
            scoreList = sorted(scoreDict.items(),key = lambda x:x[1],reverse = True)
            scoreList = scoreList[:k_shot]
            for ind, _ in scoreList:
                k_shot_Index.append(ind)
            all_index.append(k_shot_Index)
    return all_index

# according to bert embedding calculate the relevance
'''def relevance(mode, input_dir, k_shot):
    trainPath = os.path.join(input_dir, 'train.json')
    relationList = []# all relation's set, every element relation is a list    
    all_index = [] #all fact's the most related k_shot index  
    with open(trainPath) as f1:
        for line in f1.readlines():
            res = set()
            embList = []
            line = json.loads(line)
            relation = list(line['inGraph']['g_edge_types'].values())
            for r in relation:
                r = r.split('/')[-1]
                res.add(r)
            bert_embedding = BertEmbedding(model='bert_24_1024_16', dataset_name='book_corpus_wiki_en_cased')
            result = bert_embedding(list(res)) #result is a list, the element of list is a tuple, include  (token, token embedding)
            for _, emb in result:
                emb = np.array(emb)
                emb = np.sum(emb, axis=0)
                embList.append(emb)
            relationList.append(embList)
        l = len(relationList)
    if mode=='train':
        for index in range(l):
            query_re = np.array(relationList[index]) #query_re is the index's relation embedding
            scoreDict = dict() 
            k_shot_Index = [] # top k_shot revelance index in raw training data
            for i, reList in enumerate(relationList):
                if i!=index:
                    reList = np.array(reList)
                    score = cosine_similarity(query_re, reList)#score is an array
                    if 1 in score:
                        k_shot_Index.append(i)
                    else:
                        avg_score = np.average(score, axis=1)
                        final_score = np.sum(avg_score) #final_score is a float, the final score
                        scoreDict[i] = final_score
                
            if len(k_shot_Index) < k_shot:
                num = k_shot - len(k_shot_Index)
                scoreList = sorted(scoreDict.items(),key = lambda x:x[1],reverse = True)
                scoreList = scoreList[:num]
                for ind, _ in scoreList:
                    k_shot_Index.append(ind)
                all_index.append(k_shot_Index)
            
    if mode =='test':
        testPath = os.path.join(input_dir, mode+'.json')
        test_relationList = []
        with open(testPath) as f2:
            for line in f2.readlines():
                res = set()
                embList = []
                line = json.loads(line)
                relation = list(line['inGraph']['g_edge_types'].values())
                for r in relation:
                    r = r.split('/')[-1]
                    res.add(r)
                bert_embedding = BertEmbedding(model='bert_24_1024_16', dataset_name='book_corpus_wiki_en_cased')
                result = bert_embedding(list(res)) #result is a list, the element of list is a tuple, include  (token, token embedding)
                for _, emb in result:
                    emb = np.array(emb)
                    emb = np.sum(emb, axis=0)
                    embList.append(emb)
                test_relationList.append(embList)
        for index in range(len(test_relationList)):
            query_re = np.array(test_relationList[index])
            scoreDict = dict() 
            k_shot_Index = []
            for i, reList in enumerate(relationList):            
                reList = np.array(reList)
                score = cosine_similarity(query_re, reList)#score is an array
                if 1 in score:
                    k_shot_Index.append(i)
                else:
                    avg_score = np.average(score, axis=1)
                    final_score = np.sum(avg_score) #final_score is a float, the final score
                    scoreDict[i] = final_score
            if len(k_shot_Index) < k_shot:
                num = k_shot - len(k_shot_Index)
                scoreList = sorted(scoreDict.items(),key = lambda x:x[1],reverse = True)
                scoreList = scoreList[:num]
                for ind, _ in scoreList:
                    k_shot_Index.append(ind)
                all_index.append(k_shot_Index)
    return all_index'''

# select top k_shot relevance index, it is the  raw training data's index
'''def relevance(mode, input_dir,k_shot):
    trainPath = os.path.join(input_dir, 'train.json')
    relationList = []# all relation's set, every element relation is a list
    all_index = []
    with open(trainPath) as f1:
        for line in f1.readlines():
            res = set()
            line = json.loads(line)
            relation = list(line['inGraph']['g_edge_types'].values())
            for r in relation:
                r = r.split('/')[-1]
                res.add(r)
            relationList.append(list(res))
        l = len(relationList)
    if mode=='train':
        for index in range(l):
            query_re = np.array(relationList[index])
            count = 0
            k_shot_Index = []
            for i, reList in enumerate(relationList):
                if i!=index:
                    reList = np.array(reList)
                    flag = [ri in query_re for ri in reList]
                    if True in flag:
                        k_shot_Index.append(i)
                        count+=1
                    if count == k_shot:
                        break
            if count < k_shot:
                len1 = k_shot - count
                selected = []
                while len(selected)<len1:
                    selected.append(choice([ci for ci in range(0, l) if ci not in [index]]))
                np.array(selected)
                for j in selected:
                    k_shot_Index.append(j)
            all_index.append(k_shot_Index)
    if mode =='test':
        testPath = os.path.join(input_dir, mode+'.json')
        test_relationList = []
        with open(testPath) as f2:
            for line in f2.readlines():
                res = set()
                line = json.loads(line)
                relation = list(line['inGraph']['g_edge_types'].values())
                for r in relation:
                    r = r.split('/')[-1]
                    res.add(r)
                test_relationList.append(list(res))
        for index in range(l):
            query_re = np.array(test_relationList[index])
            count = 0
            k_shot_Index = []
            for i, reList in enumerate(relationList):
                reList = np.array(reList)
                flag = [ri in query_re for ri in reList]
                if True in flag:
                    k_shot_Index.append(i)
                    count+=1
                if count == k_shot:
                    break
            if count < k_shot:
                len1 = k_shot - count            
                selected = np.random.choice(l, len1, False)
                np.random.shuffle(selected)           
                for j in selected:
                    k_shot_Index.append(j)   
            all_index.append(k_shot_Index)
    return all_index'''
# according to two relation stage
'''def relevance(mode, input_dir, k_shot):
    trainPath = os.path.join(input_dir, 'train.json')
    relationList = []# all relation's set, every element relation is a list
    all_index = [] #all fact's the most related k_shot index
    bert_embedding = BertEmbedding(model='bert_24_1024_16', dataset_name='book_corpus_wiki_en_cased')  
    with open(trainPath) as f1:
        for line in f1.readlines():
            relation_Answer_based = []
            line = json.loads(line)            
            answer_ids = line['answer_ids'] #answer_ids is a list
            g_adj = line['inGraph']['g_adj'] #g_adj is a dict list
            for answer_id in answer_ids:
                rt = []
                embList = []
                flag = False                
                for k, v in g_adj.items():
                    v_node = list(v.keys())                    
                    if answer_id in v_node:
                        flag = True
                        rt.append(v[answer_id].split('/')[-1])                       
                        for k_in, v_in in g_adj.items():
                            vin_key = list(v_in.keys())
                            if k in vin_key:
                                rt.append(v_in[k].split('/')[-1]) 
                                break  # we find a two stage relation
                    if(flag):
                        break            
                if len(rt)==0:
                    p_flag = True
                    for k_1, v_1 in g_adj.items():
                        if answer_id == k_1:
                            p_flag = False
                            rt_list = list(v_1.values())
                            for rr in rt_list:
                                rt.append(rr.split('/')[-1])
                            break
                    if(p_flag):
                        res = set()
                        p_relation = list(line['inGraph']['g_edge_types'].values())
                        for p_r in p_relation:
                            p_r = p_r.split('/')[-1]
                            res.add(r)
                        rt = list(res)
                rt_embedding = bert_embedding(rt)
                for _, emb in rt_embedding:
                    emb = np.array(emb)
                    emb = np.sum(emb, axis=0)
                    embList.append(emb)#emb is embedding array
                relation_Answer_based.append(embList)
            
            relationList.append(relation_Answer_based) 
        l = len(relationList)
        # with open('./relationList.txt', 'w+') as f3:
        #     for i, reList in enumerate(relationList):
        #         f3.write(str(i)+":")
        #         for reli in reList:
        #             f3.write(reli)                    
        #         f3.write("\n")

    if mode=='train':
        for index in range(l):
            query_re = relationList[index] #query_re is the index's relation embedding
            scoreDict = dict() 
            for i, reList in enumerate(relationList):
                if i!=index:
                    # reList = np.array(reList)
                    if len(query_re)==0 or len(reList)==0:
                        rel_num_score = 0.001                        
                    else:
                        rel_num_score = 1-abs(len(query_re)-len(reList))/max(len(query_re),len(reList))
                    cos_score = []
                    for q_r in query_re:
                        q_r = np.array(q_r)
                        temp_score = []
                        for sel_r in reList:
                            sel_r = np.array(sel_r)
                            score = cosine_similarity(q_r, sel_r)#score is an array
                            max_score = np.max(score, axis=1)
                            temp_score.append(np.sum(max_score))
                        if(len(temp_score)==0):
                            temp_score.append(0.000001)
                        cos_score.append(max(temp_score))
                    cos_final_score = np.sum(np.array(cos_score))                    
                    scoreDict[i] = rel_num_score*cos_final_score
            k_shot_Index = [] # top k_shot revelance index in raw training data    
            scoreList = sorted(scoreDict.items(),key = lambda x:x[1],reverse = True)
            scoreList = scoreList[:k_shot]
            for ind, _ in scoreList:
                k_shot_Index.append(ind)
            all_index.append(k_shot_Index)
    if mode =='test':
        testPath = os.path.join(input_dir, mode+'.json')
        test_relationList = []
        with open(testPath) as f2:
            for line in f2.readlines():
                relation_Answer_based = []
                line = json.loads(line)            
                answer_ids = line['answer_ids'] #answer_ids is a list
                g_adj = line['inGraph']['g_adj'] #g_adj is a dict list
                for answer_id in answer_ids:
                    rt = []
                    embList = []
                    flag = False
                    for k, v in g_adj.items():
                        v_node = list(v.keys())                    
                        if answer_id in v_node:
                            rt.append(v[answer_id].split('/')[-1])  
                            flag = True                     
                            for k_in, v_in in g_adj.items():
                                vin_key = list(v_in.keys())
                                if k in vin_key:
                                    rt.append(v_in[k].split('/')[-1]) 
                                    break  # we find a two stage relation
                        if(flag):
                            break
                    if len(rt) ==0:
                        p_flag = True
                        for k_2, v_2 in g_adj.items():
                            if answer_id ==k_2:
                                p_flag = False
                                rt_list = list(v_2.values())
                                for rr in rt_list:
                                    rt.append(rr.split('/')[-1])
                                break
                        if(p_flag):
                            res = set()
                            p_relation = list(line['inGraph']['g_edge_types'].values())
                            for p_r in p_relation:
                                p_r = p_r.split('/')[-1]
                                res.add(r)
                            rt = list(res)
                    rt_embedding = bert_embedding(rt)
                    for _, emb in rt_embedding:
                        emb = np.array(emb)
                        emb = np.sum(emb, axis=0)
                        embList.append(emb)#emb is embedding array
                    relation_Answer_based.append(embList)
                test_relationList.append(relation_Answer_based) 
        # with open('./relationList.txt', 'w+') as f4:
        #     for i, reList in enumerate(test_relationList):
        #         f4.write(str(i)+":")
        #         for reli in reList:
        #             f4.write(reli) 
        #         f4.write("\n")       
        test_len = len(test_relationList)
        for index in range(test_len):
            # query_re = np.array(test_relationList[index])  
            query_re = test_relationList[index]      
            scoreDict = dict() 
            for i, reList in enumerate(relationList):                
                # reList = np.array(reList)
                if len(query_re)==0 or len(reList)==0:
                    rel_num_score = 0.001 
                else:
                    rel_num_score = 1-abs(len(query_re)-len(reList))/max(len(query_re),len(reList))
                cos_score = []
                for q_r in query_re:
                    q_r = np.array(q_r)
                    temp_score = []
                    for sel_r in reList:
                        sel_r = np.array(sel_r)
                        score = cosine_similarity(q_r, sel_r)#score is an array
                        max_score = np.max(score, axis=1)
                        temp_score.append(np.sum(max_score))
                    if(len(temp_score)==0):
                        temp_score.append(0.000001)
                    cos_score.append(max(temp_score))
                cos_final_score = np.sum(np.array(cos_score))                    
                scoreDict[i] = rel_num_score*cos_final_score
           
            k_shot_Index = []
            scoreList = sorted(scoreDict.items(),key = lambda x:x[1],reverse = True)
            scoreList = scoreList[:k_shot]
            for ind, _ in scoreList:
                k_shot_Index.append(ind)
            all_index.append(k_shot_Index)
    return all_index'''

# generate train/test graph adj
def process_graph_adj(mode, input_dir):
    filePath = os.path.join(input_dir, mode + '.json')    
    with open(filePath) as f:
        ind = 0
        adjList = []
        nodeIdToNameIndex = dict()
        for line in f.readlines():
            line = json.loads(line)
            nodeIdNameDict = dict()            
            g_node_names = line['inGraph']['g_node_names']
            for key, _ in g_node_names.items():
                nodeIdNameDict[key] = ind
                nodeIdToNameIndex[ind] = key              
                ind += 1
            g_adj = line['inGraph']['g_adj']
            for key, value in g_adj.items():                
                for element in list(value.keys()):
                    adjElement = []
                    adjElement.append(nodeIdNameDict[key])
                    adjElement.append(nodeIdNameDict[element])
                    adjList.append(adjElement)
   
    np.savetxt('./wholeGraphData/PQ' + mode +'Graph.txt', adjList, fmt = '%d') 
       
    '''nodeFileName = './graphData/WQTestGraph_nodeId.txt'
    with open(nodeFileName, 'w') as file:
        file.write(json.dumps(nodeIdToNameIndex)) '''  

# process the graph initial feature
def process_feature(mode, input_dir):
    filePath = os.path.join(input_dir, mode + '.json')
    nodeEmbeddingList = []# all node embedding's set, every element node embedding is a list 
    bert_embedding = BertEmbedding(model='bert_24_1024_16', dataset_name='book_corpus_wiki_en_cased')
    with open(filePath) as f1:
        for line in f1.readlines():
            nodeList = []
            res = []
            embDict = dict()
            line = json.loads(line)
            relation = list(line['inGraph']['g_edge_types'].values())
            for r in relation:
                r = r.split('/')[-1]
                res.append(r)
            res.append('[MASK]')            
            result = bert_embedding(res) #result is a list, the element of list is a tuple, include  (token, token embedding)
            # print(res, type(res))
            for (_, emb), re in zip(result, res): #emb is a list, the embedding is a 1*1024 dimension
                emb = np.array(emb)
                emb = np.sum(emb, axis=0)
                embDict[re] = emb
            g_node_names = line['inGraph']['g_node_names']
            for key, _ in g_node_names.items():
                nodeList.append(key) # nodeList is the subgraph node list
            g_adj = line['inGraph']['g_adj']
            g_adj_value = list(g_adj.values()) # list element is a dict

            # deal with g_adj_value to a dict, the dict element is {node:[relation list]}
            nodeToRelationdict = dict()
            for nodeToRelation in g_adj_value:
                for k, v in nodeToRelation.items():
                    if k in list(nodeToRelationdict.keys()):
                        nodeToRelationdict[k].append(v)
                    else:
                        nodeToRelationdict[k] = []
                        nodeToRelationdict[k].append(v)

            for node in nodeList:
                relationEmb = []
                if node in list(nodeToRelationdict.keys()):
                    for r in nodeToRelationdict[node]:
                        r = r.split('/')[-1]
                        relationEmb.append(embDict[r])
                    relationEmb = np.mean(np.array(relationEmb), axis = 0)
                    nodeEmbeddingList.append(relationEmb)
                else:
                    nodeEmbeddingList.append(embDict['[MASK]'])
               
    nodeEmbeddingList = np.array(nodeEmbeddingList) 
    np.save('./wholeGraphData/WQTrainGraph_feature1024.npy', nodeEmbeddingList)
    print(len(nodeEmbeddingList))
    

#give the graph answer entity renumber, all entity nodes start by 0
def process_graph_answer(mode, input_dir):
    filePath = os.path.join(input_dir, mode + '.json')
    with open(filePath) as f:
        ind = 0 
        graphId = 0
        answerGraphToIdNumber = dict()
        for line in f.readlines():            
            line = json.loads(line)
            nodeIdNameDict = dict()  
            answer_ids = line['answer_ids']
            # answers = line['answers']
            # answer_ids = []
            g_node_names = line['inGraph']['g_node_names'] 
            # for ans in answers:
            #     for k, v in g_node_names.items():
            #         if v == ans:
            #             answer_ids.append(k)
            #             break
            for key, value in g_node_names.items(): #nodeIdNameDict save node_name and its id_value                
                nodeIdNameDict[key] = ind                    
                ind += 1
            if len(answer_ids) == 0:
                answerGraphToIdNumber[graphId] = []
                answerGraphToIdNumber[graphId].append(ind)
            else:
                for answer_id in answer_ids:
                    if graphId in list(answerGraphToIdNumber.keys()):
                        answerGraphToIdNumber[graphId].append(nodeIdNameDict[answer_id])
                    else:   
                        answerGraphToIdNumber[graphId] = []
                        answerGraphToIdNumber[graphId].append(nodeIdNameDict[answer_id])
            graphId +=1

    nodeFileName = './wholeGraphData/WQTestGraph_ToAnswerId.txt'
    with open(nodeFileName, 'w') as file:
        file.write(json.dumps(answerGraphToIdNumber))   
    print("len:", len(answerGraphToIdNumber))    

# generate train/test each subgraph nodeID set
def process_graph_subNode(mode, input_dir):
    filePath = os.path.join(input_dir, mode + '.json')    
    with open(filePath) as f:
        ind = 0
        nodeList = []
        for line in f.readlines():
            line = json.loads(line)          
            g_node_names = line['inGraph']['g_node_names']
            temp_node = []
            for _ in list(g_node_names.keys()):                
                temp_node.append(ind)            
                ind += 1
            nodeList.append(temp_node)
    
    np.save('./dataset_qa/QA'+ mode + 'GraphNodeSet.npy', nodeList) 
 
    
# according to 2-hop relation average generate the answer embedding, and then save
def relevance(mode, input_dir):
    path = os.path.join(input_dir, mode + '.json')
    bert_embedding = BertEmbedding(model='bert_24_1024_16', dataset_name='book_corpus_wiki_en_cased')  
    with open(path) as f1:
        all_answer_emb = []
        for line in f1.readlines():
            line = json.loads(line) 
            rt = []
            embList = []
            # answer_ids = line['answer_ids'] #answer_ids is a list
            answers = line['answers']
            answer_ids = []
            g_node_names = line['inGraph']['g_node_names'] 
            g_adj = line['inGraph']['g_adj'] #g_adj is a dict list
            for ans in answers:
                for k, v in g_node_names.items():
                    if v == ans:
                        answer_ids.append(k)
                        break
            if len(answer_ids) ==0:
                rt.append('[MASK]')
            else:
                answer_id = random.choice(answer_ids) #answer_id is a str
                for k, v in g_adj.items():
                    v_node = list(v.keys())                    
                    if answer_id in v_node:                        
                        rt.append(v[answer_id][0].split('/')[-1])                       
                        for k_in, v_in in g_adj.items():
                            vin_key = list(v_in.keys())
                            if k in vin_key:
                                rt.append(v_in[k][0].split('/')[-1]) 
            if len(rt) == 0:
                rt.append('[MASK]')
            result = bert_embedding(rt) #result is a list, the element of list is a tuple, include  (token, token embedding)
            for _, emb in result:
                emb = np.array(emb)
                emb = np.sum(emb, axis=0)
                embList.append(emb)
            embList = np.average(np.array(embList), axis = 0)
            all_answer_emb.append(embList)
    np.save('./wholeDCIemb/PQ'+mode + '2hop_emb1024.npy', np.array(all_answer_emb))
        
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--k_shot', type=int, help='k shot for support set', default=5)
    argparser.add_argument('--input_dir', required=True)
    # argparser.add_argument('--output_dir', required=True)
    argparser.add_argument('--mode', default = 'train')
    args = argparser.parse_args()

    #graphToAnswerId
    # process_graph_answer(args.mode, args.input_dir)

    #generate adj
    # process_graph_adj(args.mode, args.input_dir)
    
    # read json file
    # with open('./graphData/wQTestGraph_nodeId.txt') as file1: 
    #     result = json.load(file1)
        
    #generate entity features
    # process_feature(args.mode, args.input_dir)

    #generate subgraph node set
    process_graph_subNode(args.mode, args.input_dir)

    # train_nodeSet = np.load('./dataset_qa/QAtrainGraphNodeSet.npy', allow_pickle=True)
    # print(type(train_nodeSet), train_nodeSet.shape)
    # generate 2 hop relation answerEmb
    # relevance(args.mode, args.input_dir)

    # generate the relevence top-N index 
    '''all_index = relevance(args.mode, args.input_dir, args.k_shot)
    with open(os.path.join(args.output_dir, 'relevance_two_relation_'+str(args.k_shot)+'_'+str(args.mode)+'.pickle'), 'wb') as fileWrite:
        pickle.dump(all_index, fileWrite)'''
    