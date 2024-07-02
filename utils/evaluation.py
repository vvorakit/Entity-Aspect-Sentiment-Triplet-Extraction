
import string as st

def count_same_items(dict1, dict2):
    count = 0
    for key in dict1:
        if key in dict2:
            common_items = set(dict1[key]) & set(dict2[key])
            count += len(common_items)
    return count

def extract_tag(tokens, labels):
    count_dict = {} 
    for i in range(len(tokens)):
        if labels[i] not in count_dict:
            count_dict[labels[i]] = 1
        else: 
            count_dict[labels[i]] += 1
    sorted_labels = sorted(count_dict.keys(), key=lambda label: count_dict[label], reverse=True)
    matching_key = None
    for key in count_dict.keys():
        if key == sorted_labels[0]:
            matching_key = key
            break
    return matching_key

def remove_punctuation(tokens): 
    punctuation_set = set(st.punctuation)
    punctuation_indices = []

    for i, string in enumerate(tokens):
        if all(char in punctuation_set for char in string):
            punctuation_indices.append(i)
    tokens_cleaned = [tokens[i] for i in range(len(tokens)) if i not in punctuation_indices]
    return tokens_cleaned

def merge_tokens_into_word(sentence, labels, offsets, token_position):
    initial_token_position = token_position 
    tokens_to_label, labels_for_tokens = [], []
    tokens_to_label.append(sentence[token_position])
    labels_for_tokens.append(labels[token_position])
    full_word = sentence[token_position] + sentence[token_position+1]
    token_position+=1
    tokens_to_label.append(sentence[token_position])
    labels_for_tokens.append(labels[token_position])
    while offsets[token_position][1] == offsets[token_position+1][0]: 
        full_word += sentence[token_position+1]
        token_position += 1
        tokens_to_label.append(sentence[token_position])
        labels_for_tokens.append(labels[token_position])
    word_tag = extract_tag(remove_punctuation(tokens_to_label), labels_for_tokens)
    word = ('').join(remove_punctuation(tokens_to_label))
    return word, word_tag, token_position - initial_token_position

def combine_toknes_and_labels(preds, from_to, tokens, offsets): 
    dict_labels = {}
    for item in from_to:
        start = item[0]
        help_counter = start
        end = item[1]
        if start == 0 and end == 0: 
            dict_labels['NULL'] = [preds[0]]
            continue
        target_term = [] 
        target_label = []
        for i in range(start, end+1): 
            if help_counter > end:
                break
            if offsets[help_counter][1] == offsets[help_counter+1][0]: 
                term, term_label, help_i = merge_tokens_into_word(tokens, preds, offsets, help_counter)
                target_term.append(term)
                target_label.append(term_label)
                help_counter += help_i + 1
            else: 
                target_term.append(tokens[help_counter])
                target_label.append(preds[help_counter])
                help_counter += 1
        merged_term = (' ').join(target_term)
        if merged_term not in dict_labels.keys(): 
            dict_labels[merged_term] = [extract_tag(target_term, target_label)]
        else: 
            dict_labels[merged_term].extend(extract_tag(target_term, target_label))
    return dict_labels     

def get_metrics_with_target(y_true, y_pred):
        total_pred = 0
        total_gt = 0
        total_correct = 0
        for gt, pred in zip(y_true, y_pred):
            if_empty = False
            gt_dict = {}
            pred_dict = {} 
            gt_list = gt.split(', ')
            pred_list = pred.split(', ')
            total_pred+=len(pred_list)
            total_gt+=len(gt_list)
            for item in gt_list: 
                if item == '': 
                    if_empty = True
                    continue
                splitted_item = item.split(':')
                if len(splitted_item) != 4: 
                    continue
                if splitted_item[0] not in gt_dict.keys():
                    gt_dict[splitted_item[0]] = [(':').join([splitted_item[1], splitted_item[2]])]
                else: 
                    gt_dict[splitted_item[0]].append((':').join([splitted_item[1], splitted_item[2]]))
            for item in pred_list: 
                if item == '' and if_empty: 
                    total_correct += 1
                    continue
                splitted_item = item.split(':')
                if len(splitted_item) != 4: 
                    continue
                if splitted_item[0] not in pred_dict.keys():
                    pred_dict[splitted_item[0]] = [(':').join([splitted_item[1], splitted_item[2]])]
                else: 
                    pred_dict[splitted_item[0]].append((':').join([splitted_item[1], splitted_item[2]]))
            total_correct += count_same_items(gt_dict, pred_dict)
        p = total_correct/total_pred
        r = total_correct/total_gt
        return p, r, 2*p*r/(p+r), None
                