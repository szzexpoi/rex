import json
import os
import time
import re
import argparse

parser = argparse.ArgumentParser(description="Post processing for explanation generation")
parser.add_argument("--data", type=str, default='./data', help="path to preprocessed data")
parser.add_argument("--question", type=str, default='./data', help="path to GQA questions and scene graphs")
parser.add_argument("--save", type=str, default='./', help="path for saving the data")
args = parser.parse_args()

def main():
    special_token = {'1293422':'sidewalk','2109828':'fence','1749668':'table','846582':'helmet','mice':'mouse'}

    word_dict = dict()
    for split in ['train','val']:
        explanation = json.load(open(os.path.join(args.data,'processed_explanation_'+split+'.json')))
        for qid in explanation:
            cur_exp = explanation[qid].replace('?','')
            cur_exp = [cur for cur in cur_exp.split(' ') if cur not in ['',' ']]
            for cur_word in cur_exp:
                if '(' in cur_word:
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

    rel_pool = []
    start = time.time()
    for split in ['train','val']:
        explanation = json.load(open(os.path.join(args.data,'processed_explanation_'+split+'.json')))
        question = json.load(open(os.path.join(args.question,'question',split+'_balanced_questions.json')))
        processed_data = dict()
        for idx,qid in enumerate(explanation):
            if 'ERROR' in explanation[qid] or '(obj*:scene)' in explanation[qid] or explanation[qid] =='':
                # processed_data[qid] = ''
                continue
            cur_exp = explanation[qid].replace('?','')

            # fix duplicated works (plural)
            cur_exp = ' '.join([cur for cur in cur_exp.split(' ') if cur not in ['',' ']])
            cur_exp = ' '+ cur_exp + ' '
            for k in duplicated_word:
                cur_exp = cur_exp.replace(' '+k+' ',' '+duplicated_word[k]+' ')
            cur_exp = cur_exp[1:-1]

            # fix missing 'is'
            if 'to the left of' in cur_exp and not ' is ' in cur_exp:
                cur_exp = cur_exp.split('to the left of')[0] + 'is ' + 'to the left of' +  cur_exp.split('to the left of')[1]
            elif 'to the right of' in cur_exp and not ' is ' in cur_exp:
                cur_exp = cur_exp.split('to the right of')[0] + 'is ' + 'to the right of' +  cur_exp.split('to the right of')[1]

            # fix adjective related to position
            pos_pool = {'is left': 'is on the left', 'is right': 'is on the right', 'is top': 'is at the top', 'is bottom': 'is at the bottom'}
            for cur_pos in pos_pool:
                cur_exp = cur_exp.replace(cur_pos,pos_pool[cur_pos])

            cur_exp = cur_exp.split(' ')

            # fix reversed description about position
            if ' '.join(cur_exp[-5:]) in ['is to the left of', 'is to the right of']:
                cur_exp = correct_pos(cur_exp, 1, question[qid]['answer'])
            elif ' '.join(cur_exp[-2:]) in ['is behind', 'is above', 'is below']:
                cur_exp = correct_pos(cur_exp, 2, question[qid]['answer'])
            elif ' '.join(cur_exp[-4:]) in ['is in front of', 'is on top of']:
                cur_exp = correct_pos(cur_exp, 3, question[qid]['answer'])


            # fix reversed description about relationship (e.g., xxx that xxx is wearing)
            if cur_exp[-1][-3:] == 'ing' and cur_exp[-1] != 'building' and cur_exp[-2] == 'is':
                if cur_exp[-1] not in rel_pool:
                    rel_pool.append(cur_exp[-1])
                tmp_idx = 0
                count_obj = 0
                flag = True
                for i in range(len(cur_exp)):
                    if '(' in cur_exp[i]:
                        count_obj += 1 # count the number of grounded objects
                    if cur_exp[i] == 'that' and flag:
                        tmp_idx = i # find the first "that" as the pivot for reversing the phrases
                        flag = False

                if count_obj == 2 and 'that' in cur_exp:
                    cur_exp = cur_exp[tmp_idx+1:] + cur_exp[:tmp_idx]

            cur_exp = ' '.join(cur_exp)
            # fix issues related to not(xxx)
            if 'not(' in cur_exp:
                not_attr = re.findall(r'not\(([^)]+)\)', cur_exp)
                if len(not_attr) == 1: # only 0.003% samples have more than 1 "not(" attribute thus ignore them
                    not_attr = not_attr[0]
                    # find the grounded object corresponding to the attribute
                    cur_obj =  cur_exp.split('not('+not_attr+') ')[-1].split(' ')[0]

                    # remove 'not(' and add the corresponding description at the end of explanation
                    cur_exp = cur_exp.replace('not('+not_attr+')','')
                    if not 'is' in cur_exp:
                        cur_exp = 'there is ' + cur_exp
                    cur_exp = cur_exp + ', and ' + cur_obj + ' is not ' + not_attr

            processed_data[qid] = ' '.join([cur for cur in cur_exp.split(' ') if cur not in ['',' ']])

            if (idx+1)%50000 == 0:
                print('Finished %d out of %d questions in %s split, time spent: %.2f' %(idx+1, len(explanation),split,time.time()-start))
        with open(os.path.join(args.data,'processed_explanation_'+split+'.json'),'w') as f:
            json.dump(processed_data,f)


def correct_pos(cur_exp,cur_type,answer):
    # fix reversed description about position (type 1: left/right, type 2: behind/above/below, type 3: in front of/on top of)
    reverse_dict = {'is to the left of':'is to the right of', 'is to the right of':'is to the left of', 'is behind':'is in front of',
                    'is above':'is below','is below':'is above','is in front of':'is behind','is on top of':'is below'}

    offset_dict = {1:5,2:2,3:4} # length of description about relative positions
    offset = offset_dict[cur_type]

    tmp_idx = 0
    count_obj = 0
    flag = True
    for i in range(len(cur_exp)):
        if '(' in cur_exp[i]:
            count_obj += 1 # count the number of grounded objects
        if cur_exp[i] == 'that' and flag:
            tmp_idx = i # find the first "that" as the pivot for reversing the phrases
            flag = False

    if count_obj == 2:
        if answer == 'yes':
            # in this case, we need to reverse the order of grounded objects to make it consistent with their referal order in the question
            cur_exp = cur_exp[tmp_idx+1:] + cur_exp[:tmp_idx]
        else:
            # in this case, we need to reverse the relative locations grounded objects
            pos = ' '.join(cur_exp[-offset:])
            reverse_pos = reverse_dict[pos].split(' ')
            cur_exp = cur_exp[:tmp_idx]+reverse_pos+cur_exp[tmp_idx+1:-offset]
    else:
        if answer in ['yes','no']:
            # in this case, we need to reverse the relative locations of grounded objects
            pos = ' '.join(cur_exp[-offset:])
            reverse_pos = reverse_dict[pos].split(' ')
            cur_exp = cur_exp[:tmp_idx]+reverse_pos+cur_exp[tmp_idx+1:-offset]
        else:
            # in this case, we simply add an "is" after the first grounded object (appeared at location 1)
            cur_exp = [cur_exp[0]] + ['is'] + cur_exp[1:]
    return cur_exp


main()
