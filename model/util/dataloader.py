import numpy as np
import random
import os
import time
import operator
import torch
import torch.utils.data as data
import json
import gc
from transformers import BertTokenizer


# convert problematic answer
ANS_CONVERT = {
    "a man": "man",
    "the man": "man",
    "a woman": "woman",
    "the woman": "woman",
    'one': '1',
    'two': '2',
    'three': '3',
    'four': '4',
    'five': '5',
    'six': '6',
    'seven': '7',
    'eight': '8',
    'nine': '9',
    'ten': '10',
    'grey': 'gray',
}

class Batch_generator(data.Dataset):
    def __init__(self,img_dir,que_dir,lang_dir,max_qlen=18,max_exp_len=12,seq_len=20,mode='train',percentage=100):
        self.mode = mode
        self.img_dir = img_dir
        self.max_qlen = max_qlen # maximum len of question
        self.max_exp_len = max_exp_len
        self.seq_len = seq_len
        # selecting top answers
        self.ans2idx = json.load(open(os.path.join(lang_dir,'ans2idx.json')))
        self.exp2idx = json.load(open(os.path.join(lang_dir,'exp2idx.json')))
        if self.mode == 'train' and percentage != 100:
            self.question = json.load(open(os.path.join(que_dir,'train_balanced_questions_'+str(percentage)+'.json')))
        else:
            self.question = json.load(open(os.path.join(que_dir,mode+'_balanced_questions.json')))

        if self.mode != 'testdev':
            self.explanation = json.load(open(os.path.join(lang_dir,'converted_explanation_'+mode+'.json')))
        else:
            self.explanation = dict()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.init_data()

    def init_data(self,):
        self.Q = []
        self.answer = []
        self.Img = []
        self.Qid  = []
        self.pure_answer = dict()
        self.converted_exp = []
        self.structure_gate = []

        for qid in self.question.keys():
            # convert question
            cur_Q = self.question[qid]['question'].replace(',',' ').replace('.','')
            cur_Q = [cur for cur in cur_Q.split(' ') if cur not in ['',' ']]
            cur_Q = ' '.join(cur_Q)

            if len(cur_Q.split(' '))>self.max_qlen and self.mode == 'train': #remove questions that exceed specific length, originally 14
                continue

            cur_img = self.question[qid]['imageId']

            cur_A = self.question[qid]['answer']
            if cur_A in ANS_CONVERT:
                cur_A = ANS_CONVERT[cur_A]
            self.pure_answer[qid] = cur_A
            if cur_A in self.ans2idx:
                cur_A = self.ans2idx[cur_A]

            if self.mode == 'train':
                if qid in self.explanation:
                    raw_exp = self.explanation[qid].replace('?','').replace(',',' ').replace('.','').split(' ')
                else:
                    raw_exp = ''
                raw_exp = [self.exp2idx[cur] for cur in raw_exp if cur not in ['',' ']]
                structure_gate = [0 if cur in range(1,37) else 1 for cur in raw_exp]
                if len(raw_exp)>self.max_exp_len:
                    continue
                self.converted_exp.append(raw_exp)
                self.structure_gate.append(structure_gate)

            if self.mode != 'train' and qid not in self.explanation:
                self.explanation[qid] = ''


            self.Q.append(cur_Q)
            self.answer.append(cur_A)
            self.Img.append(cur_img)
            self.Qid.append(qid)


    def eval_qa_score(self,pred,qid_list):
        acc = []
        gt = []
        for i, qid in enumerate(qid_list):
            cur_gt = self.pure_answer[qid]
            if cur_gt == pred[i]:
                acc.append(1)
            else:
                acc.append(0)
            gt.append(cur_gt)
        return acc, gt


    def __getitem__(self,index):
        question = self.Q[index]
        answer = self.answer[index]
        img_id = self.Img[index]
        qid = self.Qid[index]

        # merging question and explanation mask for inputs
        encode_question = self.tokenizer(question,return_tensors='pt')
        text_input = encode_question['input_ids'][0]
        token_type = encode_question['token_type_ids'][0]
        attention_mask = encode_question['attention_mask'][0]


        text_input = text_input[:self.seq_len]
        token_type = token_type[:self.seq_len]
        attention_mask = attention_mask[:self.seq_len]

        # padding
        pad_len = 0
        if len(text_input)<self.seq_len:
            pad_len = self.seq_len-len(text_input)
            pad_input = torch.zeros(pad_len)
            pad_token = torch.zeros(pad_len)
            pad_att_mask = torch.zeros(pad_len)
            text_input = torch.cat((text_input,pad_input),dim=0)
            token_type = torch.cat((token_type,pad_token),dim=0)
            attention_mask = torch.cat((attention_mask,pad_att_mask),dim=0)


        # load image features
        img = np.load(os.path.join(self.img_dir,str(img_id)+'.npy'))


        if self.mode == 'train':
            converted_ans = torch.zeros(len(self.ans2idx),)
            converted_ans[answer] = 1
            exp = self.converted_exp[index]
            converted_exp = torch.zeros(self.max_exp_len,len(self.exp2idx)+1)
            valid_mask = torch.zeros(self.max_exp_len)
            structure_gate = self.structure_gate[index]
            structure_gate = structure_gate[:self.max_exp_len]
            converted_gate = np.ones([self.max_exp_len,]).astype('float32')

            for i in range(len(exp)):
                converted_exp[i,exp[i]] = 1
                converted_gate[i] = structure_gate[i]
            valid_mask[:len(exp)+1] = 1

            if len(exp)<self.max_exp_len:
                converted_exp[len(exp),0] = 1

            return img, text_input.long(), token_type.long(), attention_mask.long(), converted_ans, converted_exp, valid_mask, converted_gate
        else:
            return img, text_input.long(), token_type.long(), attention_mask.long(), self.Qid[index]


    def __len__(self,):
        return len(self.Img)


class Batch_generator_transfer(data.Dataset):
    def __init__(self,img_dir,que_dir,lang_dir,max_qlen=18,max_exp_len=12,seq_len=20,mode='train',percentage=100,anno='exp'):
        self.mode = mode
        self.img_dir = img_dir
        self.max_qlen = max_qlen # maximum len of question
        self.max_exp_len = max_exp_len
        self.seq_len = seq_len
        self.anno = anno
        # selecting top answers
        self.ans2idx = json.load(open(os.path.join(lang_dir,'ans2idx.json')))
        self.exp2idx = json.load(open(os.path.join(lang_dir,'exp2idx.json')))

        if self.mode == 'train' and self.anno == 'vqa':
            self.question = json.load(open(os.path.join(que_dir,'train_balanced_questions_'+str(percentage)+'.json')))
        else:
            self.question = json.load(open(os.path.join(que_dir,mode+'_balanced_questions.json')))

        if self.mode != 'testdev':
            self.explanation = json.load(open(os.path.join(lang_dir,'converted_explanation_'+mode+'.json')))
        else:
            self.explanation = dict()

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.init_data()

    def init_data(self,):
        self.Q = []
        self.answer = []
        self.Img = []
        self.Qid  = []
        self.pure_answer = dict()
        self.converted_exp = []
        self.structure_gate = []

        for qid in self.question.keys():
            # convert question
            cur_Q = self.question[qid]['question'].replace(',',' ').replace('.','')
            cur_Q = [cur for cur in cur_Q.split(' ') if cur not in ['',' ']]
            cur_Q = ' '.join(cur_Q)

            if len(cur_Q.split(' '))>self.max_qlen and self.mode == 'train': #remove questions that exceed specific length, originally 14
                continue

            cur_img = self.question[qid]['imageId']

            cur_A = self.question[qid]['answer']
            if cur_A in ANS_CONVERT:
                cur_A = ANS_CONVERT[cur_A]
            self.pure_answer[qid] = cur_A
            if cur_A in self.ans2idx:
                cur_A = self.ans2idx[cur_A]

            if self.mode == 'train':
                raw_exp = self.explanation[qid].replace('?','').replace(',',' ').replace('.','').split(' ')
                raw_exp = [self.exp2idx[cur] for cur in raw_exp if cur not in ['',' ']]
                structure_gate = [0 if cur in range(1,37) else 1 for cur in raw_exp]
                if len(raw_exp)>self.max_exp_len:
                    continue
                self.converted_exp.append(raw_exp)
                self.structure_gate.append(structure_gate)

            if self.mode != 'train' and qid not in self.explanation:
                self.explanation[qid] = ''


            self.Q.append(cur_Q)
            self.answer.append(cur_A)
            self.Img.append(cur_img)
            self.Qid.append(qid)


    def eval_qa_score(self,pred,qid_list):
        acc = []
        for i, qid in enumerate(qid_list):
            cur_gt = self.pure_answer[qid]
            if cur_gt == pred[i]:
                acc.append(1)
            else:
                acc.append(0)
        return acc


    def __getitem__(self,index):
        question = self.Q[index]
        answer = self.answer[index]
        img_id = self.Img[index]
        qid = self.Qid[index]

        # merging question and explanation mask for inputs
        encode_question = self.tokenizer(question,return_tensors='pt')
        text_input = encode_question['input_ids'][0]
        token_type = encode_question['token_type_ids'][0]
        attention_mask = encode_question['attention_mask'][0]


        text_input = text_input[:self.seq_len]
        token_type = token_type[:self.seq_len]
        attention_mask = attention_mask[:self.seq_len]

        # padding
        pad_len = 0
        if len(text_input)<self.seq_len:
            pad_len = self.seq_len-len(text_input)
            pad_input = torch.zeros(pad_len)
            pad_token = torch.zeros(pad_len)
            pad_att_mask = torch.zeros(pad_len)
            text_input = torch.cat((text_input,pad_input),dim=0)
            token_type = torch.cat((token_type,pad_token),dim=0)
            attention_mask = torch.cat((attention_mask,pad_att_mask),dim=0)


        # load image features
        img = np.load(os.path.join(self.img_dir,str(img_id)+'.npy'))


        if self.mode == 'train':
            converted_ans = torch.zeros(len(self.ans2idx),)
            converted_ans[answer] = 1

            exp = self.converted_exp[index]
            converted_exp = torch.zeros(self.max_exp_len,len(self.exp2idx)+1)
            valid_mask = torch.zeros(self.max_exp_len)
            structure_gate = self.structure_gate[index]
            structure_gate = structure_gate[:self.max_exp_len]
            converted_gate = np.ones([self.max_exp_len,]).astype('float32')

            for i in range(len(exp)):
                converted_exp[i,exp[i]] = 1
                converted_gate[i] = structure_gate[i]
            valid_mask[:len(exp)+1] = 1

            if len(exp)<self.max_exp_len:
                converted_exp[len(exp),0] = 1
            if self.anno == 'exp':
                return img, text_input.long(), token_type.long(), attention_mask.long(), converted_exp, valid_mask, converted_gate
            elif self.anno == 'vqa':
                return img, text_input.long(), token_type.long(), attention_mask.long(), converted_ans, converted_exp, valid_mask, converted_gate

        else:
            return img, text_input.long(), token_type.long(), attention_mask.long(), self.Qid[index]


    def __len__(self,):
        return len(self.Img)


class Batch_generator_submission(data.Dataset):
    def __init__(self,img_dir,que_dir,lang_dir,seq_len=20):
        self.img_dir = img_dir
        self.seq_len = seq_len
        # selecting top answers
        self.ans2idx = json.load(open(os.path.join(lang_dir,'ans2idx.json')))
        self.exp2idx = json.load(open(os.path.join(lang_dir,'exp2idx.json')))

        self.question = json.load(open(os.path.join(que_dir,'submission_all_questions.json')))
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.init_data()

    def init_data(self,):
        self.Q = []
        self.Img = []
        self.Qid  = []

        for qid in self.question.keys():
            # convert question
            cur_Q = self.question[qid]['question'].replace(',',' ').replace('.','')
            cur_Q = [cur for cur in cur_Q.split(' ') if cur not in ['',' ']]
            cur_Q = ' '.join(cur_Q)

            cur_img = self.question[qid]['imageId']

            self.Q.append(cur_Q)
            self.Img.append(cur_img)
            self.Qid.append(qid)

    def __getitem__(self,index):
        question = self.Q[index]
        img_id = self.Img[index]
        qid = self.Qid[index]

        encode_question = self.tokenizer(question,return_tensors='pt')
        text_input = encode_question['input_ids'][0]
        token_type = encode_question['token_type_ids'][0]
        attention_mask = encode_question['attention_mask'][0]

        text_input = text_input[:self.seq_len]
        token_type = token_type[:self.seq_len]
        attention_mask = attention_mask[:self.seq_len]

        # padding
        pad_len = 0
        if len(text_input)<self.seq_len:
            pad_len = self.seq_len-len(text_input)
            pad_input = torch.zeros(pad_len)
            pad_token = torch.zeros(pad_len)
            pad_att_mask = torch.zeros(pad_len)
            text_input = torch.cat((text_input,pad_input),dim=0)
            token_type = torch.cat((token_type,pad_token),dim=0)
            attention_mask = torch.cat((attention_mask,pad_att_mask),dim=0)


        # load image features
        img = np.load(os.path.join(self.img_dir,str(img_id)+'.npy'))

        return img, text_input.long(), token_type.long(), attention_mask.long(), self.Qid[index]


    def __len__(self,):
        return len(self.Img)



#convert words within string to index
def convert_idx(sentence,word2idx):
    idx = []
    for word in sentence:
        if word in word2idx:
            idx.append(word2idx[word])
        else:
            idx.append(word2idx['UNK'])

    return idx
