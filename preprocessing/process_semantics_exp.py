import json
import numpy as np
import os
import operator
import re
import argparse

parser = argparse.ArgumentParser(description="Abstracting semantics from GQA annotations")
parser.add_argument("--question", type=str, required=True, help="path to GQA question")
parser.add_argument("--mapping", type=str, default='./data', help="path to semantic mapping")
parser.add_argument("--save", type=str, default='./data', help="path for saving the data")
args = parser.parse_args()

# abstracting the semantics for training/validation split of GQA
for split in ['train','val']:
    question = json.load(open(os.path.join(args.question,split+'_balanced_questions.json')))
    simplified_mapping = json.load(open(os.path.join(args.mapping,'simplified_mapping_exp.json')))

    processed_semantic = dict()
    for qid in question.keys():
        img_id = question[qid]['imageId']
        semantics = question[qid]['semantic']
        processed_semantic[qid] = []
        invalid_count = 0
        for i in range(len(semantics)):
            cur_semantic = semantics[i]
            if cur_semantic['operation'] == 'exist':
                processed_semantic[qid].append(('exist',-1,cur_semantic['dependencies'],-1))
                continue # redundant operation
            simplified_structure = simplified_mapping[cur_semantic['operation']]
            cur_op = simplified_structure['operation']
            # processing the first argument independently
            if simplified_structure['rel/attr'] == 'attr':
                cur_arg = cur_semantic['argument']
                cur_obj_1 = cur_semantic['dependencies']
                cur_obj_2 = -1
            elif simplified_structure['rel/attr'] == 'rel':
                cur_arg = cur_semantic['argument'].split(',')[1]
                if (re.search(r'\(([^)]+)\)', cur_semantic['argument']) is not None) and ((re.search(r'\(([^)]+)\)', cur_semantic['argument']).group(1)).split(',')[0].isdigit()):
                    cur_obj_1 = 'obj:'+ re.search(r'\(([^)]+)\)', cur_semantic['argument']).group(1)
                else:
                    cur_obj_1 = 'obj*:' + cur_semantic['argument'].split(',')[0]
                cur_obj_2 = cur_semantic['dependencies']
            elif simplified_structure['obj_1'] == 'obj':
                cur_arg = -1
                if (re.search(r'\(([^)]+)\)', cur_semantic['argument']) is not None) and ((re.search(r'\(([^)]+)\)', cur_semantic['argument']).group(1)).split(',')[0].isdigit()):
                    cur_obj_1 = 'obj:'+ re.search(r'\(([^)]+)\)', cur_semantic['argument']).group(1)
                else:
                    cur_obj_1 = 'obj*:' + cur_semantic['argument'].split(',')[0]
                cur_obj_2 = -1
            else: # the two arguments are entangled together
                if simplified_structure['obj_1'] == 'dep' and simplified_structure['obj_2'] == 'dep':
                    cur_arg = simplified_structure['rel/attr']
                    cur_obj_1 = [cur_semantic['dependencies'][0]]
                    cur_obj_2 = [cur_semantic['dependencies'][1]]
                else:
                    cur_arg = simplified_structure['rel/attr']
                    cur_obj_1 = cur_semantic['dependencies']
                    cur_obj_2 = -1
            processed_semantic[qid].append((cur_op,cur_arg,cur_obj_1,cur_obj_2))

    with open(os.path.join(args.save,'simplified_semantics_'+split+'.json'),'w') as f:
        json.dump(processed_semantic,f)
