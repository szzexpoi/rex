import json
import numpy as np
import argparse
import os
from program_executor import program_executor

parser = argparse.ArgumentParser(description="Generating explanation based on reasoning process")
parser.add_argument("--data", type=str, default='./data', help="path to preprocessed data")
parser.add_argument("--question", type=str, default='./data', help="path to GQA questions and scene graphs")
parser.add_argument("--save", type=str, default='./', help="path for saving the data")
args = parser.parse_args()

template = json.load(open(os.path.join(args.data,'exp_template.json')))
for cur_split in ['train','val']:
    processed_data = json.load(open(os.path.join(args.data,'simplified_semantics_'+cur_split+'.json')))
    question = json.load(open(os.path.join(args.question,'question',cur_split+'_balanced_questions.json')))
    scene_graph = json.load(open(os.path.join(args.question,'sceneGraphs',cur_split+'_sceneGraphs.json')))
    attr_mapping = json.load(open(os.path.join(args.data,'attr_mapping.json')))
    gen_exp = dict()
    for qid in processed_data:

        cur_semantic = processed_data[qid]
        cur_scene_graph = scene_graph[question[qid]['imageId']]
        exp = program_executor(cur_scene_graph,cur_semantic,template,attr_mapping)
        # use hard coding to fix some boundary cases
        exp = exp.replace('there is no there is no','there is no').replace('(-)','')
        exp = exp.replace('there is neither there is no','there is neither')
        exp = exp.replace('nor there is no','nor')
        if '(obj*:direction)' in exp:
            continue

        gen_exp[qid] = exp
    with open(os.path.join(args.save,'processed_explanation_'+cur_split+'.json'),'w') as f:
        json.dump(gen_exp,f)
