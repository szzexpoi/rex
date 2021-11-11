import json
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Creating subset for transfer learning")
parser.add_argument("--question", type=str, default=None, help="path to GQA questions")
args = parser.parse_args()

question = json.load(open(args.question,'train_balanced_questions.json'))
question_type = dict()
for qid in question:
    cur_type = question[qid]['types']['detailed']
    if cur_type not in question_type:
        question_type[cur_type] = []
    question_type[cur_type].append(qid)

for portion in [1,5,10]:
    selected_data = []
    for cur_type in question_type:
        if len(question_type[cur_type])<=10:
            selected_data.extend(question_type[cur_type])
        else:
            cur_portion = int(len(question_type[cur_type])*portion/100)
            tmp_data = np.random.choice(question_type[cur_type],cur_portion,replace=False)
            for cur in tmp_data:
                selected_data.append(cur)
    processed_data = dict()
    for qid in selected_data:
        processed_data[qid] = question[qid]
    print('%d samples selected for %d percent of data' %(len(processed_data),portion))
    with open(os.path.join(args.question,'train_balanced_questions_'+str(portion)+'.json'),'w') as f:
        json.dump(processed_data,f)
