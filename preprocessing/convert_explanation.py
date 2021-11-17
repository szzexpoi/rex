import json
import numpy as np
import re
import os
import time
import argparse

parser = argparse.ArgumentParser(description="Converting raw explanations for modeling")
parser.add_argument("--data", type=str, default=None, help="path to preprocessed data")
parser.add_argument("--question", type=str, default=None, help="path to GQA questions and scene graphs")
parser.add_argument("--bbox", type=str, default=None, help="path to bounding box information for UpDown features")
parser.add_argument("--save", type=str, default='./', help="path for saving the data")
args = parser.parse_args()

def compute_iou(obj,bbox):
    intersect = (min(obj[2],bbox[2])-max(obj[0],bbox[0]))*(min(obj[3],bbox[3])-max(obj[1],bbox[1]))
    union = (obj[2]-obj[0])*(obj[3]-obj[1]) + (bbox[2]-bbox[0])*(bbox[3]-bbox[1]) - intersect
    return intersect/union

# use 4x4 grid to encode position
def find_pos(raw_pos):
    pos_x, pos_y = float(raw_pos.split(',')[0]), float(raw_pos.split(',')[1])
    for i in range(1,5):
        if pos_x<=i*(1/4):
            break
    for j in range(1,5):
        if pos_y<=j*(1/4):
            break
    processed_pos = (j-1)*4+i

    return processed_pos


for split in ['train','val']:
    explanation = json.load(open(os.path.join(args.data,'processed_explanation_'+split+'.json')))
    question = json.load(open(os.path.join(args.question,'question',split+'_balanced_questions.json')))
    scene_graph = json.load(open(os.path.join(args.question,'sceneGraphs',split+'_sceneGraphs.json')))

    # converting textual explanation to multi-modal explanation
    converted_explanation = dict()
    start = time.time()
    for idx, qid in enumerate(explanation):
        cur_exp = explanation[qid]

        # special case to be solved
        if '(obj*:scene)' in cur_exp:
            converted_explanation[qid] = ''
            continue

        img_id = question[qid]['imageId']
        cur_sg = scene_graph[img_id]['objects']
        cur_bbox = np.load(os.path.join(args.bbox,img_id+'.npy'))
        # find grounded objects
        ground_obj = re.findall(r'\(([^)]+)\)', cur_exp)
        filtered_obj = []
        for cur_obj in ground_obj:
            if 'obj*:' in cur_obj or 'obj' not in cur_obj:
                if ',' in cur_obj and float(cur_obj.split(',')[0])<=2: # description of position
                    pos_idx = find_pos(cur_obj)
                    cur_exp = cur_exp.replace(cur_obj,'@'+str(pos_idx))
                continue


            filtered_obj.append(cur_obj[4:])

        # correlate objects with bbox in the bottom-up features
        grounding_pair = dict()
        for cur_obj in filtered_obj:
            x_1 = cur_sg[cur_obj]['x']
            y_1 = cur_sg[cur_obj]['y']
            x_2 = cur_sg[cur_obj]['x'] + cur_sg[cur_obj]['w']
            y_2 = cur_sg[cur_obj]['y'] + cur_sg[cur_obj]['h']
            obj_pos = (x_1,y_1,x_2,y_2)

            # compute the IoU between the current object and each bbox
            best_iou = 0
            for i in range(len(cur_bbox)):
                cur_iou = compute_iou(obj_pos,cur_bbox[i])
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    grounding_pair[cur_obj] = '#'+str(i) # special token for object grounding
        # cleaning special token
        processed_exp = cur_exp.replace('obj*:','').replace('obj:','').replace('(',' ').replace(')',' ')

        # replace objects in the text with grounding positions
        for cur_obj in grounding_pair:
            processed_exp = processed_exp.replace(cur_obj,grounding_pair[cur_obj])

        # remove ''
        processed_exp = [cur for cur in processed_exp.split(' ') if cur not in ['',' ']]
        processed_exp = ' '.join(processed_exp)
        converted_explanation[qid] = processed_exp

        if (idx+1)%10000 == 0:
            print('Processed %d out of %d samples in the %s set, time spent: %.2f' %(idx+1,len(explanation),split,time.time()-start))

    with open(os.path.join(save_dir,'converted_explanation_'+split+'.json'),'w') as f:
        json.dump(converted_explanation,f)
