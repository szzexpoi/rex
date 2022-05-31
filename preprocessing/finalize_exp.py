import json
import os
import time
import argparse

parser = argparse.ArgumentParser(description="Post processing for converted explanation")
parser.add_argument("--data", type=str, default=None, help="path to preprocessed data")
parser.add_argument("--save", type=str, default='./', help="path for saving the data")
args = parser.parse_args()

special_token = {'1293422':'sidewalk','2109828':'fence','1749668':'table','846582':'helmet','mice':'mouse'}

word_dict = dict()
for split in ['train','val']:
    explanation = json.load(open(os.path.join(args.data,'converted_explanation_'+split+'.json')))
    for qid in explanation:
        cur_exp = explanation[qid].replace('?','').replace(',',' ').replace('.',' ')
        cur_exp = [cur for cur in cur_exp.split(' ') if cur not in ['',' ']]
        for cur_word in cur_exp:
            if '#' in cur_word or '@' in cur_word:
                continue
            if cur_word in special_token:
                cur_word = special_token[cur_word]
            if cur_word not in word_dict:
                word_dict[cur_word] = len(word_dict)

# convert plural nouns into singular nouns
duplicated_word = dict()
for k in word_dict:
    if k[:-1] in word_dict and k[-1] == 's':
        duplicated_word[k] = k[:-1]
for k in special_token:
    duplicated_word[k] = special_token[k]

# exclude tokens commonly used in plural forms (e.g., shaking hands, taking pictures)
del duplicated_word['pictures']
del duplicated_word['hands']
del duplicated_word['dvds']

start = time.time()
for split in ['train','val']:
    explanation = json.load(open(os.path.join(args.data,'converted_explanation_'+split+'.json')))
    processed_data = dict()
    for idx,qid in enumerate(explanation):
        if 'ERROR' in explanation[qid]:
            # processed_data[qid] = ''
            continue
        cur_exp = explanation[qid].replace('?','').replace(',',' ').replace('.',' ')
        cur_exp = ' '.join([cur for cur in cur_exp.split(' ') if cur not in ['',' ']])
        cur_exp = ' '+ cur_exp + ' '
        for k in duplicated_word:
            cur_exp = cur_exp.replace(' '+k+' ',' '+duplicated_word[k]+' ')
        processed_data[qid] = cur_exp[1:-1]

        if (idx+1)%50000 == 0:
            print('Finished %d out of %d questions in %s split, time spent: %.2f' %(idx+1, len(explanation),split,time.time()-start))

    with open(os.path.join(args.save,'converted_explanation_'+split+'.json'),'w') as f:
        json.dump(processed_data,f)
