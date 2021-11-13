import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from torch.autograd import Variable
import numpy as np
from torch.nn.utils.weight_norm import weight_norm
from transformers import BertModel, VisualBertModel

class GRU(nn.Module):
    """
    Gated Recurrent Unit without long-term memory
    """
    def __init__(self,input_size,embed_size=512):
        super(GRU,self).__init__()
        self.update_x = nn.Linear(input_size,embed_size,bias=True)
        self.update_h = nn.Linear(embed_size,embed_size,bias=True)
        self.reset_x = nn.Linear(input_size,embed_size,bias=True)
        self.reset_h = nn.Linear(embed_size,embed_size,bias=True)
        self.memory_x = nn.Linear(input_size,embed_size,bias=True)
        self.memory_h = nn.Linear(embed_size,embed_size,bias=True)

    def forward(self,x,state):
        z = torch.sigmoid(self.update_x(x) + self.update_h(state))
        r = torch.sigmoid(self.reset_x(x) + self.reset_h(state))
        mem = torch.tanh(self.memory_x(x) + self.memory_h(torch.mul(r,state)))
        state = torch.mul(1-z,state) + torch.mul(z,mem)
        return state


class VisualBert_REX(nn.Module):
    def __init__(self,num_roi=36,nb_answer=2000,nb_vocab=2000,num_step=12,use_structure=False,lang_dir=None):
        super(VisualBert_REX,self).__init__()
        self.nb_vocab = nb_vocab
        self.num_roi = num_roi
        self.nb_answer = nb_answer
        self.num_step = num_step
        self.img_size = 2048
        self.hidden_size = 768
        self.use_structure = use_structure
        base_model = VisualBertModel.from_pretrained('uclanlp/visualbert-vqa-coco-pre')

        self.embedding = base_model.embeddings
        self.bert_encoder = base_model.encoder
        self.sent_cls = nn.Linear(768,self.nb_vocab+1)
        self.ans_cls = nn.Linear(768,self.nb_answer)


        # word embedding for explanation
        self.exp_embed = nn.Embedding(num_embeddings=nb_vocab+1,embedding_dim=self.hidden_size,padding_idx=0)

        # attentive RNN
        self.att_q = nn.Linear(self.hidden_size,self.hidden_size)
        self.att_v = nn.Linear(self.img_size,self.hidden_size)
        self.att_h = nn.Linear(self.hidden_size,self.hidden_size)
        self.att = nn.Linear(self.hidden_size,1)
        self.att_rnn = GRU(3*self.hidden_size,self.hidden_size)


        # language RNN
        self.q_fc = nn.Linear(self.hidden_size,self.hidden_size)
        self.v_fc = nn.Linear(self.img_size,self.hidden_size)

        self.language_rnn = GRU(2*self.hidden_size,self.hidden_size)
        self.language_fc = nn.Linear(self.hidden_size,nb_vocab+1)
        # self.att_drop = nn.Dropout(0.2)
        # self.fc_drop = nn.Dropout(0.2)

        if self.use_structure:
            self.structure_gate = nn.Linear(self.hidden_size,1)
            self.structure_mapping = nn.Parameter(torch.load(os.path.join(lang_dir,'structure_mapping_base.pth'),requires_grad=False))


        for module in [self.embedding,self.bert_encoder]:
            for para in module.parameters():
                para.requires_grad = True # fixed pretrained or not

    def create_att_mask(self,batch,ori_mask):
        visual_mask = torch.ones(batch,self.num_roi).cuda()
        mask = torch.cat((ori_mask,visual_mask,),dim=1)
        return mask

    def init_hidden_state(self,batch):
        init_word = torch.zeros(batch,self.hidden_size).cuda()
        init_att_h = torch.zeros(batch,self.hidden_size).cuda()
        init_language_h = torch.zeros(batch,self.hidden_size).cuda()
        return init_word, init_att_h, init_language_h

    def forward(self,img,text_input,token_type,attention_mask,exp=None,ss_rate=2):
        embedding = self.embedding(input_ids=text_input,token_type_ids=token_type,visual_embeds=img)
        concat_mask = self.create_att_mask(len(embedding),attention_mask)

        # manually create attention mask for bert encoder (copy from PreTrainedModel's function)
        extended_mask = concat_mask[:,None,None,:]
        extended_mask = (1.0-extended_mask)*-10000.0

        bert_feat = self.bert_encoder(embedding,extended_mask)[0]
        visual_feat = bert_feat[:,-int(self.num_roi):,:].contiguous()
        cls_feat = bert_feat[:,0]

        # pre-computed features for attention computation
        v_att = torch.tanh(self.att_v(img))
        q_att = torch.tanh(self.att_q(cls_feat))
        q_att = q_att.view(q_att.size(0),1,-1)
        fuse_feat = torch.mul(v_att,q_att.expand_as(v_att))

        # pre-compute features for language prediction
        q_enc = torch.tanh(self.q_fc(cls_feat))

        # initialize hidden state
        prev_word, h_1, h_2 = self.init_hidden_state(len(fuse_feat))

        # loop for explanation generation
        pred_exp = []
        pred_gate = []
        pred_att = []
        x_1 = torch.cat((fuse_feat.mean(1),h_2,prev_word),dim=-1)
        for i in range(self.num_step):
            # attentive RNN
            h_1 = self.att_rnn(x_1,h_2)
            att_h = torch.tanh(self.att_h(h_1).unsqueeze(1).expand_as(fuse_feat)+fuse_feat)
            att = F.softmax(self.att(att_h),dim=1) # with dropout
            pred_att.append(att.squeeze(1))

            # use separate layers to encode the attended features
            att_x = torch.bmm(att.transpose(1,2).contiguous(),img).squeeze()
            v_enc = torch.tanh(self.v_fc(att_x))
            fuse_enc = torch.mul(v_enc,q_enc)

            x_2 = torch.cat((fuse_enc,h_1),dim=-1)

            # language RNN
            h_2 = self.language_rnn(x_2,h_2)
            pred_word = F.softmax(self.language_fc(h_2),dim=-1) # without dropout

            if self.use_structure:
                structure_gate = torch.sigmoid(self.structure_gate(h_2))
                similarity = F.softmax(torch.bmm(h_2.unsqueeze(1),visual_feat.transpose(1,2)).squeeze(1),dim=-1)
                structure_mapping = self.structure_mapping.unsqueeze(0).expand(len(similarity),self.nb_vocab+1,self.num_roi)
                sim_pred = torch.bmm(structure_mapping,similarity.unsqueeze(-1)).squeeze(-1)
                pred_word = structure_gate*pred_word + (1-structure_gate)*sim_pred
                pred_gate.append(structure_gate)

            pred_exp.append(pred_word)

            # schedule sampling
            ss_prob = torch.rand(1)
            if ss_prob > (1-ss_rate):
                prev_word = torch.max(pred_word,dim=-1)[1]
            else:
                prev_word = torch.max(exp[:,i],dim=-1)[1]
            prev_word = self.exp_embed(prev_word)
            x_1 = torch.cat((fuse_feat.sum(1),h_2,prev_word),dim=-1)

        output_sent = torch.cat([_.unsqueeze(1) for _ in pred_exp],dim=1)
        output_ans = F.softmax(self.ans_cls(cls_feat),dim=-1)
        output_att = torch.cat([_.unsqueeze(1) for _ in pred_att],dim=1)

        if self.use_structure:
            output_gate = torch.cat([_ for _ in pred_gate],dim=1)
            return output_ans, output_sent, output_gate
        else:
            return output_ans, output_sent


class VisualBert_REX_transfer(nn.Module):
    def __init__(self,num_roi=36,nb_answer=2000,nb_vocab=2000,num_step=12,use_structure=False,anno_type='exp',lang_dir=None):
        super(VisualBert_REX_transfer,self).__init__()
        self.nb_vocab = nb_vocab
        self.num_roi = num_roi
        self.nb_answer = nb_answer
        self.num_step = num_step
        self.img_size = 2048
        self.hidden_size = 768
        self.use_structure = use_structure
        self.anno_type = anno_type
        base_model = VisualBertModel.from_pretrained('uclanlp/visualbert-vqa-coco-pre')

        self.embedding = base_model.embeddings
        self.bert_encoder = base_model.encoder
        self.sent_cls = nn.Linear(768,self.nb_vocab+1)

        # word embedding for explanation
        self.exp_embed = nn.Embedding(num_embeddings=nb_vocab+1,embedding_dim=self.hidden_size,padding_idx=0)

        # attentive RNN
        self.att_q = nn.Linear(self.hidden_size,self.hidden_size)
        self.att_v = nn.Linear(self.img_size,self.hidden_size)
        self.att_h = nn.Linear(self.hidden_size,self.hidden_size)
        self.att = nn.Linear(self.hidden_size,1)
        self.att_rnn = GRU(3*self.hidden_size,self.hidden_size)


        # language RNN
        self.q_fc = nn.Linear(self.hidden_size,self.hidden_size)
        self.v_fc = nn.Linear(self.img_size,self.hidden_size)

        self.language_rnn = GRU(2*self.hidden_size,self.hidden_size)
        self.language_fc = nn.Linear(self.hidden_size,nb_vocab+1)
        # self.att_drop = nn.Dropout(0.2)
        # self.fc_drop = nn.Dropout(0.2)

        if self.use_structure:
            self.structure_gate = nn.Linear(self.hidden_size,1)
            self.structure_mapping = nn.Parameter(torch.load(os.path.join(lang_dir,'structure_mapping_base.pth'),requires_grad=False))


        for module in [self.embedding,self.bert_encoder]:
            for para in module.parameters():
                para.requires_grad = True # fixed pretrained or not

        if self.anno_type == 'vqa':
            self.ans_cls = nn.Linear(768,self.nb_answer)


    def create_att_mask(self,batch,ori_mask):
        visual_mask = torch.ones(batch,self.num_roi).cuda()
        mask = torch.cat((ori_mask,visual_mask,),dim=1)
        return mask

    def init_hidden_state(self,batch):
        init_word = torch.zeros(batch,self.hidden_size).cuda()
        init_att_h = torch.zeros(batch,self.hidden_size).cuda()
        init_language_h = torch.zeros(batch,self.hidden_size).cuda()
        return init_word, init_att_h, init_language_h

    def forward(self,img,text_input,token_type,attention_mask,exp=None,ss_rate=2):
        embedding = self.embedding(input_ids=text_input,token_type_ids=token_type,visual_embeds=img)
        concat_mask = self.create_att_mask(len(embedding),attention_mask)

        # manually create attention mask for bert encoder (copy from PreTrainedModel's function)
        extended_mask = concat_mask[:,None,None,:]
        extended_mask = (1.0-extended_mask)*-10000.0

        bert_feat = self.bert_encoder(embedding,extended_mask)[0]
        visual_feat = bert_feat[:,-int(self.num_roi):,:].contiguous()
        cls_feat = bert_feat[:,0]

        # pre-computed features for attention computation
        v_att = torch.tanh(self.att_v(img))
        q_att = torch.tanh(self.att_q(cls_feat))
        q_att = q_att.view(q_att.size(0),1,-1)
        fuse_feat = torch.mul(v_att,q_att.expand_as(v_att))

        # pre-compute features for language prediction
        q_enc = torch.tanh(self.q_fc(cls_feat))

        # initialize hidden state
        prev_word, h_1, h_2 = self.init_hidden_state(len(fuse_feat))

        # loop for explanation generation
        pred_exp = []
        pred_gate = []
        x_1 = torch.cat((fuse_feat.mean(1),h_2,prev_word),dim=-1)
        for i in range(self.num_step):
            # attentive RNN
            h_1 = self.att_rnn(x_1,h_2)
            att_h = torch.tanh(self.att_h(h_1).unsqueeze(1).expand_as(fuse_feat)+fuse_feat)
            att = F.softmax(self.att(att_h),dim=1) # with dropout

            # use separate layers to encode the attended features
            att_x = torch.bmm(att.transpose(1,2).contiguous(),img).squeeze()
            v_enc = torch.tanh(self.v_fc(att_x))
            fuse_enc = torch.mul(v_enc,q_enc)

            x_2 = torch.cat((fuse_enc,h_1),dim=-1)

            # language RNN
            h_2 = self.language_rnn(x_2,h_2)
            pred_word = F.softmax(self.language_fc(h_2),dim=-1) # without dropout

            if self.use_structure:
                structure_gate = torch.sigmoid(self.structure_gate(h_2))
                similarity = F.softmax(torch.bmm(h_2.unsqueeze(1),visual_feat.transpose(1,2)).squeeze(1),dim=-1)
                structure_mapping = self.structure_mapping.unsqueeze(0).expand(len(similarity),self.nb_vocab+1,self.num_roi)
                sim_pred = torch.bmm(structure_mapping,similarity.unsqueeze(-1)).squeeze(-1)
                pred_word = structure_gate*pred_word + (1-structure_gate)*sim_pred
                pred_gate.append(structure_gate)

            pred_exp.append(pred_word)

            # schedule sampling
            ss_prob = torch.rand(1)
            if ss_prob > (1-ss_rate):
                prev_word = torch.max(pred_word,dim=-1)[1]
            else:
                prev_word = torch.max(exp[:,i],dim=-1)[1]
            prev_word = self.exp_embed(prev_word)
            x_1 = torch.cat((fuse_feat.sum(1),h_2,prev_word),dim=-1)

        output_sent = torch.cat([_.unsqueeze(1) for _ in pred_exp],dim=1)

        if self.use_structure:
            output_gate = torch.cat([_ for _ in pred_gate],dim=1)
            if self.anno_type == 'vqa':
                output_ans = F.softmax(self.ans_cls(cls_feat),dim=-1)
                return output_ans, output_sent, output_gate
            else:
                return output_sent, output_gate

        else:
            if self.anno_type == 'vqa':
                output_ans = F.softmax(self.ans_cls(cls_feat),dim=-1)
                return output_ans, output_sent
            else:
                return output_sent
