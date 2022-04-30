import  torch
import os
import  numpy as np
from dataset import collate
from dataset import Dataset
from    torch.utils.data import DataLoader
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse
from meta import Meta
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"

def main():
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)
    flag = 1 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    maml = Meta(args, flag, device)
    maml = maml.to(device)# rewrite our's model, the "config" rewrite ours bart pretrained model path.
  
    train_pt = os.path.join(args.output_dir, 'train.pt')
    data = Dataset (train_pt, args.input_dir, args.output_dir, mode = 'train', taskNum = 18989, k_shot = args.k_spt, k_query=1) #WQ
    test_pt = os.path.join(args.output_dir, 'test.pt')
    data_test = Dataset (test_pt, args.input_dir, args.output_dir, mode = 'test', taskNum = 2000, k_shot = args.k_spt, k_query=1) #WQ
    db = DataLoader(data, args.task_num, shuffle=True, num_workers=16, pin_memory=True, collate_fn = collate)#num_workers and pin_memory is trying to speed up 
    db_test = DataLoader(data_test, 1, shuffle=False, num_workers=16, pin_memory=True, collate_fn = collate)
    
    print("start!!!!")
    for epoch in range(args.epoch):
        for step, (x_spt,x_qry, query_index) in enumerate(db):
            x_spt, x_qry, query_index = x_spt.to(device), x_qry.to(device), query_index.to(device)
            bleu_1, bleu_2, bleu_3, bleu_4, rouge_l = maml(x_spt, x_qry, query_index)
            if (step+1) % 30 == 0:
                print('step:', step, '\ttraining bleu1-4, rouge_l:', bleu_1, bleu_2, bleu_3, bleu_4, rouge_l)  
           
        accs_all_test_1 = []
        accs_all_test_2 = []
        accs_all_test_3 = []
        accs_all_test_4 = []
        accs_all_test_r = []
        loss_all_test = []
        bestPred = []
        for x_spt, x_qry, query_index in db_test:
            x_spt, x_qry, query_index = x_spt.to(device), x_qry.to(device), query_index.to(device)
            bleu_1, bleu_2, bleu_3, bleu_4, rouge_l, loss, predMaxQ = maml.finetunning(x_spt,  x_qry, query_index, 'test')
            accs_all_test_1.append(bleu_1)
            accs_all_test_2.append(bleu_2)
            accs_all_test_3.append(bleu_3)
            accs_all_test_4.append(bleu_4)
            accs_all_test_r.append(rouge_l)
            loss_all_test.append(loss)
            bestPred.append(predMaxQ)
        # [b, update_step+1]
        # bleu = np.array(accs_all_test).mean(axis=0).astype(np.float16)
        test_bleu_1 = np.array(accs_all_test_1).max(axis = 1).mean(axis=0).astype(np.float16)
        test_bleu_2 = np.array(accs_all_test_2).max(axis = 1).mean(axis=0).astype(np.float16)
        test_bleu_3 = np.array(accs_all_test_3).max(axis = 1).mean(axis=0).astype(np.float16)
        test_bleu_4 = np.array(accs_all_test_4).max(axis = 1).mean(axis=0).astype(np.float16)
        test_bleu_r = np.array(accs_all_test_r).max(axis = 1).mean(axis=0).astype(np.float16)
        test_loss = np.array(loss_all_test).min(axis = 1).mean(axis=0).astype(np.float16)
        
        #save the best generated question
        bestPred = np.array(bestPred)
        np.savetxt(os.path.join(args.output_dir, 'predict_maml_WQ_' + str(epoch)+'.txt'), bestPred, fmt='%s')
        
        with open(os.path.join(args.input_dir, 'Test_bleu_maml_WQ.txt'), 'a') as ft:
            ft.write(str(epoch)+" 1-4 and rouge-l:" +str(test_bleu_1)+'\t'+str(test_bleu_2)+'\t'+ str(test_bleu_3)+'\t'+str(test_bleu_4)+'\t'+str(test_bleu_r))
            ft.write('\n')
       
        print('Test bleu :',test_bleu_1, test_bleu_2, test_bleu_3, test_bleu_4, test_bleu_r, test_loss)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=20)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=20)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=20)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=8)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=3e-5)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=5e-5)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=1)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=5)
    argparser.add_argument('--model_name_or_path', required = True) # bart model path
    argparser.add_argument('--weight_decay', default=1e-5, type=float)
    argparser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    argparser.add_argument('--input_dir', required=True)
    argparser.add_argument('--output_dir', required=True)
    argparser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    args = argparser.parse_args()

    main()
