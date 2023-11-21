import json
import numpy as np
import re
import os
import time
import operator
import torch
import argparse

parser = argparse.ArgumentParser(description="Generating dictionary for model training")
parser.add_argument("--question", type=str, default=None, help="path to GQA question files")
parser.add_argument("--exp", type=str, default=None, help="path to converted explanations")
parser.add_argument("--save", type=str, default='./', help="path for saving the data")
args = parser.parse_args()

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


exp_dict = dict()
ans_dict = dict()

# insert special tokens for visual grounding
for i in range(36):
	exp_dict['#'+str(i)] = len(exp_dict)+1 # indices for grounding
for i in range(1,17):
	exp_dict['@'+str(i)] = len(exp_dict)+1 # indices for position encoding

exp_len = dict()
for split in ['train','val']:
	explanation = json.load(open(os.path.join(args.exp,'converted_explanation_'+split+'.json')))
	question = json.load(open(os.path.join(args.question,split+'_balanced_questions.json')))
	for qid in explanation:
		cur_exp = explanation[qid].replace('?','').replace(',',' ').replace('.','')
		cur_ans = question[qid]['answer']
		exp_word = [cur for cur in cur_exp.split(' ') if cur not in ['',' ']]

		if cur_ans in ANS_CONVERT:
			cur_ans = ANS_CONVERT[cur_ans]

		if cur_ans not in ans_dict:
			ans_dict[cur_ans] = len(ans_dict)

		if len(exp_word) not in exp_len:
			exp_len[len(exp_word)] = 0
		exp_len[len(exp_word)] += 1

		for cur_word in exp_word:
			if cur_word not in exp_dict:
				exp_dict[cur_word] = len(exp_dict) + 1

# print statistics
print('Number of answers: %d' %len(ans_dict))
print('Number of vocabularies for explanation: %d' %len(exp_dict))

# create structure mapping for REX
structure_mapping = torch.zeros(len(exp_dict)+1,36) # mapping words to 36 input regions (UpDown features)
for i in range(36):
	structure_mapping[exp_dict['#'+str(i)],i] = 1

with open(os.path.join(args.save,'exp2idx.json'),'w') as f:
	json.dump(exp_dict,f)
with open(os.path.join(args.save,'ans2idx.json'),'w') as f:
	json.dump(ans_dict,f)
torch.save(structure_mapping,os.path.join(args.save,'structure_mapping.pth'))
