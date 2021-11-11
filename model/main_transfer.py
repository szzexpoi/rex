import sys
sys.path.append('./util')
sys.path.append('./model')
sys.path.append('pycocoevalcap')
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from dataloader import Batch_generator_transfer
from evaluation import organize_eval_data, construct_sentence, Grounding_Evaluator, Attribute_Evaluator
from bert_exp import VisualBert_REGEX_transfer
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
import numpy as np
import cv2
import argparse
import os
import time
import gc
import tensorflow as tf
from loss import cross_entropy, ans_cross_entropy, structure_bce
import json
from eval_exp import COCOEvalCap


parser = argparse.ArgumentParser(description='Transfer learning experiment')
parser.add_argument('--mode', type=str, default='train', help='Selecting running mode (default: train)')
parser.add_argument('--anno_dir',type=str, default=None, help='Directory to annotations')
parser.add_argument('--sg_dir',type=str, default=None, help='Directory to scene graph')
parser.add_argument('--ood_dir',type=str, default=None, help='Directory to annotations')
parser.add_argument('--lang_dir',type=str, default='./processed_lang', help='Directory to preprocessed language files')
parser.add_argument('--img_dir',type=str, default=None, help='Directory to image features')
parser.add_argument('--bbox_dir',type=str, default=None, help='Directory to bounding box information')
parser.add_argument('--checkpoint_dir',type=str, default=None, help='Directory for saving checkpoint')
parser.add_argument('--weights',type=str, default=None, help='Trained model to be loaded (default: None)')
parser.add_argument('--epoch',type=int, default=8, help='Defining maximal number of epochs')
parser.add_argument('--lr',type=float, default=4e-4, help='Defining initial learning rate (default: 4e-4)')
parser.add_argument('--batch_size',type=int, default=128, help='Defining batch size for training (default: 150)')
parser.add_argument('--word_size',type=int, default=300, help='Defining size for word embedding (default: 300)')
parser.add_argument('--embedding_size',type=int, default=1024, help='Defining size for embedding (default: 1024)')
parser.add_argument('--hidden_size',type=int, default=1024, help='Defining size for embedding in explanation generator (default: 1024)')
parser.add_argument('--clip',type=float, default=0.1, help='Gradient clipping to prevent gradient explode (default: 0.1)')
parser.add_argument('--max_qlen',type=int, default=18, help='Maximum length of question')
parser.add_argument('--max_exp_len',type=int, default=12, help='Maximum length of explanation')
parser.add_argument('--seq_len',type=int, default=20, help='Sequence length after padding')
parser.add_argument('--alpha',type=float, default=1, help='Balance factor for sentence loss')
parser.add_argument('--beta',type=float, default=1, help='Balance factor for structure gate loss')
parser.add_argument('--use_structure',type=bool, default=False, help='use the structure gate or not')
parser.add_argument('--percentage',type=int, default=100, help='percentage of training data')
parser.add_argument('--anno_type',type=str, default='exp', help='annotation type for training')

args = parser.parse_args()



def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_learning_rate(init_lr,optimizer, epoch):
    "adatively adjust lr based on epoch"
    if args.anno_type == 'vqa':
        lr = init_lr * (0.25 ** int((epoch+1)/12)) #lr decay for vqa transfer learning
    else:
        lr = init_lr * (0.25 ** int((epoch+1)/8)) # lr decay for exp generation pretraining

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main_exp():
    tf_summary_writer = tf.summary.create_file_writer(args.checkpoint_dir)

    train_data = Batch_generator_transfer(args.img_dir,args.anno_dir,args.lang_dir,args.max_qlen,args.max_exp_len,args.seq_len,'train',args.percentage,args.anno_type)
    val_data = Batch_generator_transfer(args.img_dir,args.anno_dir,args.lang_dir,30,args.max_exp_len,35,'val',args.percentage,args.anno_type)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=12)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=12)
    exp2idx = train_data.exp2idx

    idx2exp = dict()
    for k in exp2idx:
        idx2exp[exp2idx[k]] = k

    # initialize evaluator for visual grounding
    grounding_evaluator = Grounding_Evaluator(args.lang_dir,args.bbox,args.sg_dir)

    # initialize evaluator for attributes
    attribute_evaluator = Attribute_Evaluator(args.lang_dir)


    model = VisualBert_REGEX_transfer(nb_vocab=len(exp2idx),num_step=args.max_exp_len,use_structure=args.use_structure,anno_type=args.anno_type)
    model = nn.DataParallel(model)
    model = model.cuda()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0) #1e-8
    # optimizer = optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()))

    def train(iteration):
        model.train()
        avg_sent_loss = 0
        avg_structure_loss = 0

        for batch_idx,(img, text_input, token_type, attention_mask, exp, valid_mask, structure_gate) in enumerate(trainloader):
            if len(img)<args.batch_size:
                continue
            img, text_input, token_type, attention_mask, exp, valid_mask, structure_gate = Variable(img), Variable(text_input), Variable(token_type), Variable(attention_mask), Variable(exp), Variable(valid_mask), Variable(structure_gate)
            img, text_input, token_type, attention_mask, exp, valid_mask, structure_gate = img.cuda(), text_input.cuda(), token_type.cuda(), attention_mask.cuda(), exp.cuda(), valid_mask.cuda(), structure_gate.cuda()
            optimizer.zero_grad()

            if args.use_structure:
                pred, pred_structure = model(img,text_input, token_type, attention_mask,exp,ss_rate=0)
                exp_loss = args.alpha*cross_entropy(pred,exp,valid_mask)
                structure_loss = args.beta*structure_bce(pred_structure,structure_gate)
                loss = exp_loss + structure_loss

            else:
                pred = model(img,text_input, token_type, attention_mask,exp,ss_rate=0)
                exp_loss = args.alpha*cross_entropy(pred,exp,valid_mask)
                loss = exp_loss
            loss.backward()

            if not args.clip == 0 :
                clip_grad_norm_(model.parameters(),args.clip)
            optimizer.step()
            avg_sent_loss = (avg_sent_loss*np.maximum(0,batch_idx) + exp_loss.data.cpu().numpy())/(batch_idx+1)

            if args.use_structure:
                avg_structure_loss = (avg_structure_loss*np.maximum(0,batch_idx) + structure_loss.data.cpu().numpy())/(batch_idx+1)

            if batch_idx%25 == 0:
                with tf_summary_writer.as_default():
                    tf.summary.scalar('Sentence Reconstruction loss',avg_sent_loss,step=iteration)
                    if args.use_structure:
                        tf.summary.scalar('Structure prediction loss',avg_structure_loss,step=iteration)

            iteration += 1

        return iteration

    def test(iteration):
        model.eval()
        res = []
        gt = []
        qid_list = []

        for batch_idx,(img,text_input, token_type, attention_mask, qid) in enumerate(valloader):
            img, text_input, token_type, attention_mask = Variable(img), Variable(text_input), Variable(token_type), Variable(attention_mask)
            img, text_input, token_type, attention_mask = img.cuda(), text_input.cuda(), token_type.cuda(), attention_mask.cuda()

            if args.use_structure:
                pred, pred_structure = model(img,text_input, token_type, attention_mask)
            else:
                pred = model(img,text_input, token_type, attention_mask)
            pred = pred.data.cpu().numpy()

            pred_exp = construct_sentence(pred, idx2exp)

            res.extend(pred_exp)
            for idx,cur_id in enumerate(qid):
                qid_list.append(cur_id)
                gt.append(val_data.explanation[cur_id])

        grounding_score = grounding_evaluator.eval_grounding(res,qid_list)
        attr_score, record_attr = attribute_evaluator.eval_attribute(res,qid_list)

        res, gt, error_count = organize_eval_data(res,gt,qid_list)

        exp_evaluator = COCOEvalCap(gt,res)
        exp_score = exp_evaluator.evaluate()


        with tf_summary_writer.as_default():
            for metric in exp_score:
                tf.summary.scalar(metric,exp_score[metric]*len(res)/(len(res)+error_count),step=iteration)
            tf.summary.scalar('Grounding Score',grounding_score,step=iteration)
            for attribute in attr_score:
                tf.summary.scalar('Recall for attribute '+ attribute, attr_score[attribute]*100,step=iteration)

        return exp_score['CIDEr']


    #main loop for training:
    print('Start training model')
    iteration = 0
    val_score = 0
    for epoch in range(args.epoch):
        adjust_learning_rate(args.lr,optimizer, epoch)
        iteration = train(iteration)
        torch.save(model.module.state_dict(),os.path.join(args.checkpoint_dir,'model.pth'))
        cur_score = test(iteration)

        #save the best check point
        if cur_score > val_score:
            torch.save(model.module.state_dict(),os.path.join(args.checkpoint_dir,'model_best.pth'))
            val_score = cur_score



def main_vqa():
    tf_summary_writer = tf.summary.create_file_writer(args.checkpoint_dir)

    train_data = Batch_generator_transfer(args.img_dir,args.anno_dir,args.lang_dir,args.max_qlen,args.max_exp_len,args.seq_len,'train',args.percentage,args.anno_type)
    val_data = Batch_generator_transfer(args.img_dir,args.anno_dir,args.lang_dir,30,args.max_exp_len,35,'val',args.percentage,args.anno_type)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=12)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=12)
    ans2idx = train_data.ans2idx
    exp2idx = train_data.exp2idx

    ood_data = dict()
    for keyword in ['all','head','tail']:
        ood_data[keyword] = json.load(open(os.path.join(args.ood_dir,'ood_val_'+keyword+'.json')))

    # create mapping from index to word
    idx2ans = dict()
    for k in ans2idx:
        idx2ans[ans2idx[k]] = k
    idx2exp = dict()
    for k in exp2idx:
        idx2exp[exp2idx[k]] = k


    # initialize evaluator for visual grounding
    grounding_evaluator = Grounding_Evaluator(args.lang_dir,args.bbox,args.sg_dir)

    # initialize evaluator for attributes
    attribute_evaluator = Attribute_Evaluator(args.lang_dir)


    model = VisualBert_REGEX_transfer(nb_answer=len(ans2idx),nb_vocab=len(exp2idx),num_step=args.max_exp_len,use_structure=args.use_structure,anno_type=args.anno_type)
    model.load_state_dict(torch.load(args.weights),strict=False)
    model = nn.DataParallel(model)
    model = model.cuda()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0) #1e-8
    # optimizer = optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()))

    def train(iteration):
        model.train()
        avg_ans_loss = 0
        avg_sent_loss = 0
        avg_structure_loss = 0

        for batch_idx,(img, text_input, token_type, attention_mask, ans, exp, valid_mask,structure_gate) in enumerate(trainloader):
            if len(img) < args.batch_size:
                continue
            img, text_input, token_type, attention_mask, ans, exp, valid_mask, structure_gate = Variable(img), Variable(text_input), Variable(token_type), Variable(attention_mask), Variable(ans), Variable(exp), Variable(valid_mask), Variable(structure_gate)
            img, text_input, token_type, attention_mask, ans, exp, valid_mask, structure_gate = img.cuda(), text_input.cuda(), token_type.cuda(), attention_mask.cuda(), ans.cuda(), exp.cuda(), valid_mask.cuda(), structure_gate.cuda()
            optimizer.zero_grad()

            if args.use_structure:
                pred_ans, pred, pred_structure = model(img,text_input, token_type, attention_mask,exp,ss_rate=0)
                ans_loss = ans_cross_entropy(pred_ans,ans)
                exp_loss = args.alpha*cross_entropy(pred,exp,valid_mask)
                structure_loss = args.beta*structure_bce(pred_structure,structure_gate)
                loss = ans_loss + exp_loss + structure_loss

            else:
                pred_ans, pred = model(img,text_input, token_type, attention_mask,exp,ss_rate=0)
                ans_loss = ans_cross_entropy(pred_ans,ans)
                exp_loss = args.alpha*cross_entropy(pred,exp,valid_mask)
                loss = ans_loss + exp_loss
            loss.backward()

            if not args.clip == 0 :
                clip_grad_norm_(model.parameters(),args.clip)
            optimizer.step()
            avg_ans_loss = (avg_ans_loss*np.maximum(0,batch_idx) + ans_loss.data.cpu().numpy())/(batch_idx+1)
            avg_sent_loss = (avg_sent_loss*np.maximum(0,batch_idx) + exp_loss.data.cpu().numpy())/(batch_idx+1)

            if args.use_structure:
                avg_structure_loss = (avg_structure_loss*np.maximum(0,batch_idx) + structure_loss.data.cpu().numpy())/(batch_idx+1)

            if batch_idx%25 == 0:
                with tf_summary_writer.as_default():
                    tf.summary.scalar('Answer loss',avg_ans_loss,step=iteration)
                    tf.summary.scalar('Sentence Reconstruction loss',avg_sent_loss,step=iteration)
                    if args.use_structure:
                        tf.summary.scalar('Structure prediction loss',avg_structure_loss,step=iteration)

            iteration += 1

        return iteration

    def test(iteration):
        model.eval()
        res = []
        gt = []
        qid_list = []
        total_acc = []

        for batch_idx,(img,text_input, token_type, attention_mask, qid) in enumerate(valloader):
            img, text_input, token_type, attention_mask = Variable(img), Variable(text_input), Variable(token_type), Variable(attention_mask)
            img, text_input, token_type, attention_mask = img.cuda(), text_input.cuda(), token_type.cuda(), attention_mask.cuda()

            if args.use_structure:
                pred_ans, pred, pred_structure = model(img,text_input, token_type, attention_mask)
            else:
                pred_ans, pred = model(img,text_input, token_type, attention_mask)

            #computing accuracy
            pred_ans = pred_ans.data.cpu().numpy()
            pred_ans = pred_ans.argmax(-1)
            pred = pred.data.cpu().numpy()

            pred_exp = construct_sentence(pred, idx2exp)

            converted_ans = []
            res.extend(pred_exp)
            for idx,cur_id in enumerate(qid):
                qid_list.append(cur_id)
                gt.append(val_data.explanation[cur_id])
                converted_ans.append(idx2ans[pred_ans[idx]])
            ans_score = val_data.eval_qa_score(converted_ans,qid)
            total_acc.extend(ans_score)

        grounding_score, record_grounding = grounding_evaluator.eval_grounding(res,qid_list)
        attr_score, record_attr = attribute_evaluator.eval_attribute(res,qid_list)

        res, gt, error_count = organize_eval_data(res,gt,qid_list)
        exp_evaluator = COCOEvalCap(gt,res)
        exp_score = exp_evaluator.evaluate()

        # compute ood VQA accuracy
        ood_acc = dict()
        for keyword in ood_data:
            ood_acc[keyword] = 0
            for idx, qid in enumerate(qid_list):
                if qid in ood_data[keyword]:
                    ood_acc[keyword] += total_acc[idx]
            ood_acc[keyword] /= len(ood_data[keyword])

        # compute normal VQA accuracy
        total_acc = np.mean(total_acc)*100


        with tf_summary_writer.as_default():
            for metric in exp_score:
                tf.summary.scalar(metric,exp_score[metric]*len(res)/(len(res)+error_count),step=iteration)
            tf.summary.scalar('VQA accuracy',total_acc,step=iteration)
            tf.summary.scalar('Grounding Score',grounding_score,step=iteration)
            for keyword in ood_data:
                tf.summary.scalar('OOD-'+ keyword + ' accuracy',ood_acc[keyword]*100,step=iteration)
            for attribute in attr_score:
                tf.summary.scalar('Recall for attribute '+ attribute, attr_score[attribute]*100,step=iteration)

        return total_acc


    #main loop for training:
    print('Start training model')
    iteration = 0
    val_score = 0
    for epoch in range(args.epoch):
        adjust_learning_rate(args.lr,optimizer, epoch)
        iteration = train(iteration)
        torch.save(model.module.state_dict(),os.path.join(args.checkpoint_dir,'model.pth'))
        if (epoch+1)%5 ==0  or (epoch+1) ==args.epoch:
            cur_score = test(iteration)
            #save the best check point
            if cur_score > val_score:
                torch.save(model.module.state_dict(),os.path.join(args.checkpoint_dir,'model_best.pth'))
                val_score = cur_score

def evaluation():
    test_data = Batch_generator_transfer(args.img_dir,args.anno_dir,args.lang_dir,30,args.max_exp_len,35,'testdev')
    testloader = torch.utils.data.DataLoader(test_data, batch_size=30, shuffle=False, num_workers=8)

    ans2idx = test_data.ans2idx
    exp2idx = test_data.exp2idx

    ood_data = dict()
    for keyword in ['all','head','tail']:
        ood_data[keyword] = json.load(open(os.path.join(args.ood_dir,'ood_testdev_'+keyword+'.json')))


    # create mapping from index to word
    idx2ans = dict()
    for k in ans2idx:
        idx2ans[ans2idx[k]] = k

    idx2exp = dict()
    for k in exp2idx:
        idx2exp[exp2idx[k]] = k


    model = VisualBert_REGEX_transfer(nb_answer=len(ans2idx),nb_vocab=len(exp2idx),num_step=args.max_exp_len,use_structure=args.use_structure)
    model.load_state_dict(torch.load(args.weights))
    model = model.cuda()

    model.eval()
    qid_list = []
    total_acc = []

    for batch_idx,(img,text_input, token_type, attention_mask, qid) in enumerate(testloader):
        img, text_input, token_type, attention_mask = Variable(img), Variable(text_input), Variable(token_type), Variable(attention_mask)
        img, text_input, token_type, attention_mask = img.cuda(), text_input.cuda(), token_type.cuda(), attention_mask.cuda()
        if args.use_structure:
            pred_ans, pred, pred_structure = model(img,text_input, token_type, attention_mask)
        else:
            pred_ans, pred = model(img,text_input, token_type, attention_mask)

        #computing accuracy
        pred_ans = pred_ans.data.cpu().numpy()
        pred_ans = pred_ans.argmax(-1)

        converted_ans = []
        for idx,cur_id in enumerate(qid):
            qid_list.append(cur_id)
            converted_ans.append(idx2ans[pred_ans[idx]])
        ans_score = test_data.eval_qa_score(converted_ans,qid)
        total_acc.extend(ans_score)

    # compute ood VQA accuracy
    ood_acc = dict()
    for keyword in ood_data:
        ood_acc[keyword] = 0
        for idx, qid in enumerate(qid_list):
            if qid in ood_data[keyword]:
                ood_acc[keyword] += total_acc[idx]
        ood_acc[keyword] /= len(ood_data[keyword])

    # compute normal VQA accuracy
    total_acc = np.mean(total_acc)*100

    print('VQA Accuracy: %.2f' %total_acc)
    for keyword in ood_data:
        print('OOD-%s accuracy: %.5f' %(keyword,ood_acc[keyword]*100))


if args.mode == 'train' and args.anno_type == 'exp':
    main_exp()
elif args.mode == 'train' and args.anno_type == 'vqa':
    main_vqa()
elif args.mode == 'eval':
    evaluation()
else:
    assert 0, 'Invalid mode selected'
