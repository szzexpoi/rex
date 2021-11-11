import json
import re
import os
import numpy as np

def validate_obj(phrase):
    # auxiliary function for excluding attributes with '()' and selecting validate object
    obj_pool = re.findall(r'\(([^)]+)\)', phrase)
    for obj_id in obj_pool:
        if 'obj:' in obj_id:
            select_obj = obj_id[4:]
            break
    return select_obj

def count_obj(phrase):
    # auxiliary function to check the number of grounded objects in the phrase
    obj_pool = re.findall(r'\(([^)]+)\)', phrase)
    return np.sum([1 for cur in obj_pool if 'obj:' in cur])

def lcs(S,T):
    m = len(S)
    n = len(T)
    counter = [[0]*(n+1) for x in range(m+1)]
    longest = 0
    lcs_set = None
    for i in range(m):
        for j in range(n):
            if S[i] == T[j]:
                c = counter[i][j] + 1
                counter[i+1][j+1] = c
                if c > longest:
                    longest = c
                    lcs_set = S[i-c+1:i+1]

    # only consider the common substring related to object description (followed by an "is")
    idx = 0
    lcs_set = lcs_set.split(' ')
    for i, cur in enumerate(lcs_set):
        if cur == 'is':
            idx = i+1
    lcs_set = ' '.join(lcs_set[:idx])

    return lcs_set

def program_executor(scene_graph,semantic,template,attr_mapping):
    height, width = scene_graph['height'], scene_graph['width']
    scene_graph = scene_graph['objects']
    return step(scene_graph,semantic,template,-1,height,width,attr_mapping) # scene graph, original data, template, index for the step

# recursively construct the explanation through graph traversal
def step(scene_graph,semantic,template,idx,height,width,attr_mapping):
    # print('#########')
    cur_op, cur_attr, cur_obj_1, cur_obj_2 = semantic[idx]
    cur_template = template[cur_op]
    decomposed_step = cur_template.split('#')
    step_output = dict()
    # iterate every component (defined in the template) in the current step
    for i in range(len(decomposed_step)):
        step_output[i] = dict()
        if re.search(r'\(([^)]+)\)', decomposed_step[i]) is not None:
            cur_arg = re.search(r'\(([^)]+)\)', decomposed_step[i]).group(1)
            if cur_arg == 'ARG':
                if cur_attr == 'name':
                    obj_name = step(scene_graph,semantic,template,cur_obj_1[0],height,width,attr_mapping)
                    obj_name = validate_obj(obj_name)
                    cur_attr = scene_graph[obj_name]['name']
                step_output[i]['content'] = cur_attr
                step_output[i]['type'] = 'attribute'
            elif cur_arg == 'OBJ':
                if len(cur_obj_1.split(','))==1:
                    step_output[i]['content'] = '('+cur_obj_1+')'
                else:
                    cur_obj_1_pool = ['('+cur_obj_1.split(',')[0]+')']
                    for obj_idx in range(1,len(cur_obj_1.split(','))):
                        cur_obj_1_pool.append('(obj:'+cur_obj_1.split(',')[obj_idx]+')')
                    step_output[i]['content'] = ','.join(cur_obj_1_pool)
                step_output[i]['type'] = 'object' # change type from obj to dep to accommodate relate operation
            elif cur_arg == 'DEP1':
                step_output[i]['content'] = ','.join([step(scene_graph,semantic,template,cur_idx,height,width,attr_mapping) for cur_idx in cur_obj_1])
                step_output[i]['type'] = 'dependency'
            else:
                step_output[i]['content'] = step(scene_graph,semantic,template,cur_obj_2[0],height,width,attr_mapping)
                step_output[i]['type'] = 'dependency'
        elif re.search(r'\[([^)]+)\]', decomposed_step[i]) is not None:
            action = re.search(r'\[([^)]+)\]', decomposed_step[i]).group(1)
            step_output[i]['content'] = action # execute action after collecting data from all dependent nodes
            step_output[i]['type'] = 'action'
        else:
            step_output[i]['content'] = decomposed_step[i]
            step_output[i]['type'] = 'phrase'

    # check if there exists a pending action
    if 'action' not in [step_output[cur]['type'] for cur in step_output]:
        return ' '.join([step_output[cur]['content'] for cur in step_output])
    else:
        return run_action(scene_graph,step_output,cur_attr,height,width,attr_mapping)

def exist_dep(dependency):
    dependency = dependency[0]
    if 'obj*' in dependency:
        return 'there is no ' + dependency
    else:
        return dependency

    # return dependency

def find_common(scene_graph,dependency):
    # dep_1 = re.findall(r'\(([^)]+)\)', dependency[0])[0][4:]
    # dep_2 = re.findall(r'\(([^)]+)\)', dependency[1])[0][4:]
    dep_1 = validate_obj(dependency[0])
    dep_2 = validate_obj(dependency[1])
    attr_1 = scene_graph[dep_1]['attributes']
    attr_2 = scene_graph[dep_2]['attributes']
    for cur_attr in attr_1:
        if cur_attr in attr_2:
            common_attr = cur_attr
            break
    return 'both '+dependency[0]+' and '+dependency[1]+' are '+common_attr

def logic_and(dependency):
    if 'obj*' not in dependency[0] and 'obj*' not in dependency[1]:
        dep_1 = validate_obj(dependency[0])
        dep_2 = validate_obj(dependency[1])

        # use different templates to encode the following two cases
        if dep_1 == dep_2:
            # # handle cases with reference to the same object
            # num_obj = count_obj(dependency[1])
            # if num_obj ==1:
            #     return 'there is ' + dependency[0] + ' and ' + dependency[1].replace('(obj:'+dep_2+')','it')
            # else:
            #     return 'there is ' + dependency[0] + ' and ' + dependency[1]

            # handle cases with reference to the same object (using LCS)
            overlap = lcs(dependency[0],dependency[1])
            return 'there is ' + dependency[0] + ' and ' + dependency[1].replace(overlap,'')
        else:
            return 'there are ' + dependency[0] + ' and ' + dependency[1]

    elif 'obj*' not in dependency[0]:
        return 'there is no '+ dependency[1]
    elif 'obj*' not in dependency[1]:
        return 'there is no '+ dependency[0]
    else:
        return 'there is neither '+ dependency[0] + ' nor ' + dependency[1]

def logic_or(dependency):
    if 'obj*' not in dependency[0] and 'obj*' not in dependency[1]:
        dep_1 = validate_obj(dependency[0])
        dep_2 = validate_obj(dependency[1])
        if dep_1 == dep_2:
            # num_obj = count_obj(dependency[1])
            # if num_obj ==1:
            #     return 'there is ' + dependency[0] + ' and ' + dependency[1].replace('(obj:'+dep_2+')','it')
            # else:
            #     return 'there is ' + dependency[0] + ' and ' + dependency[1]

            # handle cases with reference to the same object (using LCS)
            overlap = lcs(dependency[0],dependency[1])
            return 'there is ' + dependency[0] + ' and ' + dependency[1].replace(overlap,'')
        else:
            return 'there are ' + dependency[0] + ' and ' + dependency[1]
    elif 'obj*' not in dependency[0]:
        return 'there is '+ dependency[0]
    elif 'obj*' not in dependency[1]:
        return 'there is '+ dependency[1]
    else:
        return 'there is neither '+ dependency[0] + ' nor ' + dependency[1]

def same_attr(scene_graph,select_attr,dependency,attr_mapping):
    if select_attr == 'type':
        # verifying if the dependencies are of the same species
        type_dict = dict()
        for cur_dep in dependency:
            # cur_dep = re.findall(r'\(([^)]+)\)', dependency[0])[0][4:]
            cur_dep = validate_obj(cur_dep)
            type_dict[cur_dep] = scene_graph[cur_dep]['name']

        if len(np.unique(list(type_dict.values()))) == 1:
            return ','.join([cur for cur in dependency])+' are all '+type_dict[cur_dep]
        else:
            type_dict_new = dict()
            for i, cur_id in enumerate(type_dict):
                if type_dict[cur_id] not in type_dict_new:
                    type_dict_new[type_dict[cur_id]] = []
                type_dict_new[type_dict[cur_id]].append(dependency[i])
            phrase = []
            for cur_type in type_dict_new:
                tmp_phrase = ' and '.join([cur for cur in type_dict_new[cur_type]])
                if len(type_dict_new[cur_type])>1:
                    tmp_phrase += ' are '+cur_type
                else:
                    tmp_phrase += ' is '+cur_type
                phrase.append(tmp_phrase)

            return ','.join(phrase)
    else:
        # verifying if the dependencies have the same attributes
        attr_dict = dict()
        for cur_dep in dependency:
            # cur_dep = re.findall(r'\(([^)]+)\)', dependency[0])[0][4:]
            cur_dep = validate_obj(cur_dep)
            attributes = scene_graph[cur_dep]['attributes']
            if select_attr == 'gender':
                # handle attribute related to gender
                if scene_graph[cur_dep]['name'] in ['woman','girl','lady']:
                    attributes.append('female')
                else:
                    attributes.append('male')
            attr_dict[cur_dep] = []
            for cur_attr in attributes:
                if cur_attr in attr_mapping['categorized'][select_attr]:
                    attr_dict[cur_dep].append(cur_attr)

        for cur_attr in attr_dict[cur_dep]:
            flag_same = True
            for dep in attr_dict:
                if cur_attr not in attr_dict[dep]:
                    flag_same = False
                    break
            if flag_same:
                break

        if flag_same:
            return ','.join([cur for cur in dependency])+' are all '+cur_attr
        else:
            attr_dict_new = dict()
            for i, cur_dep in enumerate(attr_dict):
                cur_attr = np.random.choice(attr_dict[cur_dep],1)[0] # in case there are more than 1 attribute found
                if cur_attr not in attr_dict_new:
                    attr_dict_new[cur_attr] = []
                attr_dict_new[cur_attr].append(dependency[i])

            phrase = []
            for cur_attr in attr_dict_new:
                tmp_phrase = ' and '.join([cur for cur in attr_dict_new[cur_attr]])
                if len(attr_dict_new[cur_attr])>1:
                    tmp_phrase += ' are '+cur_attr
                else:
                    tmp_phrase += ' is '+cur_attr
                phrase.append(tmp_phrase)
            return ','.join(phrase)

def compare_attr(scene_graph,select_attr,dependency,attr_mapping,height,width):
    # dep_1 = re.findall(r'\(([^)]+)\)', dependency[0])[0][4:]
    # dep_2 = re.findall(r'\(([^)]+)\)', dependency[1])[0][4:]
    dep_1 = validate_obj(dependency[0])
    dep_2 = validate_obj(dependency[1])

    if select_attr not in ['lower','higher']:
        # comparison irrelevant to position
        # print(select_attr)
        select_attr = attr_mapping['comparative'][select_attr]
        attr_1 = scene_graph[dep_1]['attributes'] + [scene_graph[dep_1]['name']]
        attr_2 = scene_graph[dep_2]['attributes'] + [scene_graph[dep_2]['name']]
        attr_dict = dict()
        for i, cur_attr in enumerate([attr_1,attr_2]):
            attr_dict[dependency[i]] = []
            for candidate_attr in select_attr:
                if candidate_attr in cur_attr:
                    attr_dict[dependency[i]].append(candidate_attr)
        # print(attr_1)
        # print(attr_2)
        # print(attr_dict)
        # print('#######')
        phrase = []
        for cur_dep in attr_dict:
            if len(attr_dict[cur_dep])!=0:
                phrase.append(cur_dep + ' is '+','.join(attr_dict[cur_dep]))
        return ', while '.join(phrase)
    else:
        # comparison specific to vertical position
        pos_1 = (round(scene_graph[dep_1]['x']/width,1), round(scene_graph[dep_1]['y']/height,1))
        pos_2 = (round(scene_graph[dep_2]['x']/width,1), round(scene_graph[dep_2]['y']/height,1))

        return dependency[0] +' is located at ('+str(pos_1[0])+','+str(pos_1[1])+'), while ' + dependency[1] +' is located at ('+str(pos_2[0])+','+str(pos_2[1])+')'

def verify_pos(scene_graph,select_attr,object,height,width):
    if select_attr == 'left':
        return scene_graph[object]['x']<(width/2)
    elif select_attr == 'right':
        return scene_graph[object]['x']>=(width/2)
    elif select_attr in ['bottom','lower']:
            return scene_graph[object]['y']>(height/2) # indices start from upper left
    elif select_attr in ['top','higher']:
        return scene_graph[object]['y']<=(height/2)

def query_attr(scene_graph,select_attr,dependency,height,width,attr_mapping):
    # query the value of the selected attributes for the dependency (single)
    if "obj*" in dependency[0]:
        return "there is no " + dependency[0]
    else:
        cur_dep = validate_obj(dependency[0])

    query_res = 'ERROR' # special token for debugging

    if select_attr in attr_mapping['categorized']:
        # for valid attribute
        if select_attr in ['hposition','vposition']:
            # query related to position
            pos_pool = []
            for cur_obj in dependency:
                cur_obj = validate_obj(cur_obj)
                pos_pool.append((round(scene_graph[cur_obj]['x']/width,1), round(scene_graph[cur_obj]['y']/height,1)))
            phrase = []
            for i,cur_obj in enumerate(dependency):
                tmp_phrase = cur_obj + ' is located at ('+str(pos_pool[i][0])+','+str(pos_pool[i][1])+')'
                phrase.append(tmp_phrase)
            return ', '.join(phrase)

        elif select_attr in ['gender','place','room']:
            # query related to name of objects
            query_res = scene_graph[cur_dep]['name']
        else:
            # query related to attributes of objects
            select_category = attr_mapping['categorized'][select_attr]
            for cur_attr in select_category:
                if cur_attr in scene_graph[cur_dep]['attributes']:
                    query_res = cur_attr
                    break
    elif select_attr.isdigit() or select_attr in ['None','sport']:
        # problematic annotations in GQA
        select_category = attr_mapping['categorized']['uncategorized']
        for cur_attr in select_category:
            if cur_attr in scene_graph[cur_dep]['attributes']:
                query_res = cur_attr
                break
    else:
        # query the name of the dependency
        query_res = scene_graph[cur_dep]['name']

    if len(dependency) > 1:
        return ','.join(dependency) + ' are ' + query_res
    else:
        return dependency[0] + ' is ' + query_res

def verify_attr(scene_graph,select_attr,dependency,height,width):
    if '|' not in select_attr:
        flag = [False]*len(dependency)
        # examine if the current attribute is in the format of "not (attribute)"
        if 'not(' in select_attr:
            select_attr = re.findall(r'\(([^)]+)\)', select_attr)[0]

        # remove issues related to attributes padded with " "
        select_attr = ' '.join([cur for cur in select_attr.split(' ') if cur not in ['',' ']])

        if select_attr in ['left','right','top','bottom','higher','lower']:
            # attribute being verified is related to position
            pos_pool = []
            for cur_obj in dependency:
                cur_obj = validate_obj(cur_obj)
                pos_pool.append((round(scene_graph[cur_obj]['x']/width,1), round(scene_graph[cur_obj]['y']/height,1)))
            phrase = []
            for i,cur_obj in enumerate(dependency):
                tmp_phrase = cur_obj + ' is located at ('+str(pos_pool[i][0])+','+str(pos_pool[i][1])+')'
                phrase.append(tmp_phrase)
            return ', '.join(phrase)
        else:
            # attribute being verified is irrelevant to position
            for i,cur_dep in enumerate(dependency):
                # temporal fix for non-existing object
                if 'obj*' in cur_dep:
                    return 'there is no '+dependency[i]

                cur_dep = validate_obj(cur_dep)
                cur_attr = scene_graph[cur_dep]['attributes'] + [scene_graph[cur_dep]['name']]
                flag[i] = True if select_attr in cur_attr else False


        if np.sum(flag)==len(dependency):
            # all dependencies have the desired attributes
            if len(dependency)>1:
                return ','.join(dependency)+' are '+select_attr
            else:
                return dependency[0]+' is '+select_attr
        else:
            # some dependencies fail to pass the verification
            if len(dependency) == 1:
                return dependency[0]+' is not '+select_attr
            else:
                passed_id = [idx for idx in range(len(dependency)) if flag[idx]]
                failed_id = [idx for idx in range(len(dependency)) if not flag[idx]]
                count_phrase = ' are ' if len(passed_id)>1 else ' is '
                passed_phrase = ','.join(dependency) + count_phrase + select_attr
                count_phrase = ' are not' if len(failed_id)>1 else ' is not'
                failed_phrase = ','.join(dependency) + count_phrase + select_attr
                return ', while '.join(passed_phrase,failed_phrase)
    else:
        # current attribute is in the form of A|B (A or B)
        invalid_dep = dict()
        for candidate_attr in select_attr.split('|'):
            flag = [False]*len(dependency)
            if candidate_attr in ['left','right','top','bottom','higher','lower']:
                # attribute being verified is related to position
                for i, cur_dep in enumerate(dependency):
                    # temporal fix for non-existing object
                    if 'obj*' in cur_dep:
                        return 'there is no '+dependency[i]
                    cur_dep = validate_obj(cur_dep)
                    flag[i] = verify_pos(scene_graph,candidate_attr,cur_dep,height,width)
            else:
                # attribute being verified is irrelevant to position
                for i,cur_dep in enumerate(dependency):
                    # temporal fix for non-existing object
                    if 'obj*' in cur_dep:
                        return 'there is no '+dependency[i]

                    cur_dep = validate_obj(cur_dep)
                    cur_attr = scene_graph[cur_dep]['attributes'] + [scene_graph[cur_dep]['name']]
                    flag[i] = True if candidate_attr in cur_attr else False

            # check if all dependencies have the current attribute
            if np.sum(flag)==len(dependency):
                # all dependencies have the desired attributes
                if len(dependency)>1:
                    return ','.join(dependency)+' are '+candidate_attr
                else:
                    return dependency[0]+' is '+candidate_attr


def relate_attr(scene_graph,select_attr,object,dependency):
    # verify if the object and dependency have certain relationship
    if 'obj*' in object:
        # current object does not exist in the scene
        if not '|' in select_attr:
            # return 'there is no ' + dependency[0] + ' ' + select_attr + ' ' + object # double-check
            return 'there is no ' + object + ' ' + select_attr + ' ' + dependency[0]
        else:
            # return 'there is no ' + dependency[0] + ' ' + ' or '.join(select_attr.split('|')) + ' ' + object
            return 'there is no ' + ' or '.join([object + ' ' + cur_attr + dependency[0] for cur_attr in select_attr.split('|')])

    obj_name = validate_obj(object)
    dep_name = validate_obj(dependency[0]) # for relate, there is only one dependency
    obj = scene_graph[obj_name]
    dep = scene_graph[dep_name]

    replaced_attr = None # replace ambiguous relationship "of" with a specific one

    if not '|' in select_attr:
        reverse_flag = False
        flag = False
        for i in range(len(obj['relations'])):
            # note that some questions use ambiguous relationship "of"
            if obj['relations'][i]['object'] == dep_name and (obj['relations'][i]['name'] == select_attr or select_attr == 'of'):
                flag = True
                replaced_attr = obj['relations'][i]['name']
                break

        # the relationship may be stored in another direction
        if not flag:
            for i in range(len(dep['relations'])):
                if dep['relations'][i]['object'] == obj_name and (dep['relations'][i]['name'] == select_attr or select_attr=='of'):
                    flag = True
                    reverse_flag = True
                    replaced_attr = dep['relations'][i]['name']
                    break

        if flag:
            if reverse_flag:
                if select_attr != 'of':
                    return object + ' that ' + dependency[0] + ' is ' + select_attr
                else:
                    return object + ' that ' + dependency[0] + ' is ' + replaced_attr
            else:
                if select_attr != 'of':
                    return object + ' ' + select_attr + ' ' + dependency[0]
                else:
                    return object + ' ' + replaced_attr + ' ' + dependency[0]
        else:
            if reverse_flag:
                return object + ' that ' + dependency[0] + ' is not ' + select_attr
            else:
                return object + ' is not ' + select_attr + ' ' + dependency[0]
    else:
        # choose 1 out of 2 relationships
        select_attr = select_attr.split('|')
        final_attr = None
        reverse_flag = False

        for cur_attr in select_attr:
            for i in range(len(obj['relations'])):
                if obj['relations'][i]['object'] == dep_name and obj['relations'][i]['name'] == cur_attr:
                    final_attr = cur_attr
                    break

        # dependency may be stored in the reversed order
        if final_attr is None:
            for cur_attr in select_attr:
                for i in range(len(dep['relations'])):
                    if dep['relations'][i]['object'] == obj_name and dep['relations'][i]['name'] == cur_attr:
                        final_attr = cur_attr
                        reverse_flag = True
                        break

        if final_attr is not None:
            if reverse_flag:
                return object + ' that ' + dependency[0] + ' is ' + final_attr
            else:
                return object + ' ' + final_attr + ' ' + dependency[0]

        else:
            return 'there is no ' + object + ' ' + ' or '.join(select_attr) + ' ' + dependency[0]


def run_action(scene_graph,step_output,select_attr,height,width,attr_mapping):
    dependency = []
    for step_idx in step_output:
        step = step_output[step_idx]
        if step['type'] == 'action':
            cur_action = step['content']
        elif step['type'] == 'dependency':
            for cur_dep in step['content'].split(','):
                dependency.append(cur_dep)
        elif step['type'] == 'object':
            object = step['content'] # for relate operation
    if cur_action == 'EXIST_DEP':
        return exist_dep(dependency)
    elif cur_action == 'FIND_COMMON':
        return find_common(scene_graph,dependency)
    elif cur_action in ['SAME_ATTR','DIFFERENT_ATTR']:
        return same_attr(scene_graph,select_attr,dependency,attr_mapping)
    elif cur_action == 'COMPARE_ATTR':
        return compare_attr(scene_graph,select_attr,dependency,attr_mapping,height,width)
    elif cur_action == 'LOGIC_AND':
        return logic_and(dependency)
    elif cur_action == 'LOGIC_OR':
        return logic_or(dependency)
    elif cur_action == 'VERIFY_ATTR':
        return verify_attr(scene_graph,select_attr,dependency,height,width)
    elif cur_action == 'RELATE_ATTR':
        return relate_attr(scene_graph,select_attr,object,dependency)
    elif cur_action == 'QUERY_ATTR':
        return query_attr(scene_graph,select_attr,dependency,height,width,attr_mapping)
