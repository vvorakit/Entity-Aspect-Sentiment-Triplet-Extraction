import xml.etree.ElementTree as ET
import pandas as pd
from EASTE_transformers.src.transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForMultipleChoice
import ast
import itertools

def make_str_to_list(string):
    return ast.literal_eval(str(string))

def batch_generator(iterable, batch_size=16):
    iterable = iter(iterable)
    while True:
        batch = list(itertools.islice(iterable, batch_size))
        if len(batch) > 0:
            yield batch
        else:
            break

def xml_to_dataframe(dataset_path):
    tree = ET.parse(dataset_path)
    root = tree.getroot()
    ids,texts,targets,categories,polarities,frm,tolist=[],[],[],[],[],[],[]
    for child in root:
        for c in child:
            for s in c:
                for o in s:
                    texts.append(s.find('text').text)
                    ids.append(s.attrib)
                    idl,t,c,p,f,tl=[],[],[],[],[],[]
                    for i in s.findall("./Opinions/Opinion"):
                        t.append(i.get("target"))
                        c.append(i.get("category"))
                        p.append(i.get("polarity"))
                        f.append(i.get("from"))
                        tl.append(i.get("to"))
                    targets.append(t)
                    categories.append(c)
                    polarities.append(p)
                    frm.append(f)
                    tolist.append(tl)
    tolists=[]
    idl, txt, frms, polarity, target, category = [],[],[],[],[],[]
    for i in range(len(texts)): 

        try:
            if texts[i] == texts[i+1]: 
                continue
        except IndexError:
            pass
        
        tolists.append(tolist[i])
        idl.append(ids[i])
        txt.append(texts[i])
        frms.append(frm[i])
        polarity.append(polarities[i])
        target.append(targets[i])
        category.append(categories[i])

    lst=[]
    lst.append(idl)
    lst.append(txt)
    lst.append(category)
    lst.append(target)
    lst.append(polarity)
    lst.append(frms)
    lst.append(tolists)
    df=pd.DataFrame(lst,index=["ids","texts","categories","targets","polarities","from","to"]).T
    # del_empty(df)
    return df

def get_semeval2016task5(df,model='bert'):
    if model=='multibert':
        tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')
    elif model=='camembert':
        tokenizer = AutoTokenizer.from_pretrained('camembert-base')
    elif model=='mgpt':
        tokenizer = AutoTokenizer.from_pretrained('sberbank-ai/mGPT')
    elif model=='bert':
        print('Using a default pre-trained model: BERT')
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    assigned_all_elements = []
    # process per row
    Texts=[]
    for row in df.index:
        text=str(df['texts'][int(row)])
        Texts.append(text)
        
        categories = make_str_to_list(df['categories'][int(row)])
        targets = make_str_to_list(df['targets'][int(row)])
        polarities = make_str_to_list(df['polarities'][int(row)])
        start_idxs = make_str_to_list(df['from'][int(row)])
        end_idxs = make_str_to_list(df['to'][int(row)])
        
        # to make (0,0) annotate only on the first one
        count_null = 0 
        
        # process per text in a row
        tokenized_text = tokenizer(text,return_offsets_mapping=True, padding=False, truncation=True)# padding="max_length", max_length=50,
        row_arranged_token, row_arranged_category, row_arranged_offset,row_arranged_polarity = [],[],[],[]
        count = 0
        check=0
        
        #create the list of end indexes as a reference
        check_token={}
        for input_id,attention_mask,offset_mapping in zip(tokenized_text['input_ids'],tokenized_text['attention_mask'],tokenized_text['offset_mapping']):
            end_offset_mapping = offset_mapping[1]
            check_token[end_offset_mapping]=0
        
        
        for input_id,attention_mask,offset_mapping in zip(tokenized_text['input_ids'],tokenized_text['attention_mask'],tokenized_text['offset_mapping']):
            start_offset_mapping = offset_mapping[0]
            end_offset_mapping = offset_mapping[1]

            if check_token[end_offset_mapping]==1 and end_offset_mapping!=0:
                continue

            category_idx = 0
            attempt_checks = len(start_idxs)
            checking_if_counted = 0
            l=len(end_idxs)
            
            # **attention** The list of entity#aspect regarding the customed datasets need to list here, otherwise it is annotated as 'others' 
            notothers=['RESTAURANT#PRICES', 'AMBIENCE#GENERAL', 'DRINKS#PRICES', 'SERVICE#GENERAL', 'LOCATION#GENERAL', 'FOOD#STYLE_OPTIONS', 'RESTAURANT#GENERAL', 'RESTAURANT#MISCELLANEOUS', 'FOOD#QUALITY', 'DRINKS#STYLE_OPTIONS', 'DRINKS#QUALITY', 'FOOD#PRICES']
            for i in range(l):
                if(check_token[end_offset_mapping]==0):
                    start_idx, end_idx = start_idxs[i],end_idxs[i]
                    #(0,0) case separation
                    if int(end_idx)==0 and int(end_offset_mapping)==0 and count_null==0:
                        check_token[end_offset_mapping]=1
                        count_null = 1
                        offset = (0,0)
                        token = ''
                        row_arranged_token.append(token)
                        row_arranged_offset.append(offset)
                        if categories[i] in notothers: 
                            row_arranged_category.append(categories[i])
                            row_arranged_polarity.append(polarities[i])
                        else: 
                            row_arranged_category.append('None#')
                            row_arranged_polarity.append('None#')

                        checking_if_counted=1
                        category_idx += 1
                    #all the other cases
                    elif int(start_idx) <= int(start_offset_mapping) and int(end_idx) >= int(end_offset_mapping) and end_offset_mapping!=0 :
                        offset = (start_offset_mapping,end_offset_mapping)
                        token = text[start_offset_mapping:end_offset_mapping]
                        row_arranged_token.append(token)
                        row_arranged_offset.append(offset)
                        if categories[category_idx] in notothers: 
                            row_arranged_category.append(categories[category_idx])
                            row_arranged_polarity.append(polarities[category_idx])

                        else: 
                            row_arranged_category.append('None#')
                            row_arranged_polarity.append('None#')

                        category_idx += 1
                        checking_if_counted=1
                        check_token[end_offset_mapping]=1

            if checking_if_counted==0 :
                offset = (start_offset_mapping,end_offset_mapping)
                token = text[start_offset_mapping:end_offset_mapping]
                row_arranged_token.append(token)
                row_arranged_offset.append(offset)
                row_arranged_category.append('None#')
                row_arranged_polarity.append('None#')

                check_token[end_offset_mapping]=1
        assigned_all_elements.append([row_arranged_token,row_arranged_category,row_arranged_polarity,row_arranged_offset,targets,categories,polarities,start_idxs,end_idxs])
    df_assigned_all_token = pd.DataFrame(assigned_all_elements, columns = ['tokens', 'entity_aspects','sentiment','offsets','targets','categories', 'polarities', 'from','to'])
    df_assigned_all_token['Texts']=Texts
    return df_assigned_all_token

def get_semeval2016task5_multi_annotation(train_df):
    
    train_df['len_categories'] = train_df['categories'].apply(lambda x: len(x))
    train_df_help = train_df[train_df.len_categories > 1].copy()
    train_df.drop(train_df[train_df['Texts'].isin(train_df_help['Texts'])].index, inplace = True)
    train_df.reset_index(inplace=True, drop=True)
    train_df.drop(['len_categories'], axis=1, inplace=True)
    
    for index, row in train_df_help.iterrows():
        help_row = train_df_help.loc[index].copy()
        train_df.drop(train_df[train_df['Texts'].isin([help_row['Texts']])].index, inplace = True) 
        train_df.reset_index(inplace=True, drop=True)
        no_tokens = len(help_row.tokens)
        tagging_dict = get_tagging_info(help_row)
        max_duplicates_key = max(len(v) for k,v in tagging_dict.items())
        for i in range(max_duplicates_key): 
            help_row.entity_aspects = ['None#']*no_tokens
            help_row.sentiment = ['None#']*no_tokens
            for key in tagging_dict.keys(): 
                start_index = key[0]
                end_index = key[1]
                try: 
                    entity_aspect = tagging_dict[key][i][0]
                    sentiment = tagging_dict[key][i][1]
                except IndexError: 
                    entity_aspect = tagging_dict[key][-1][0]
                    sentiment = tagging_dict[key][-1][1]
                for j in range(no_tokens): 
                    if j>=start_index and j<=end_index: 
                        help_row.entity_aspects[j] = entity_aspect
                        help_row.sentiment[j] = sentiment 
            train_df.loc[len(train_df)] = help_row
    return train_df  

def get_tagging_info(pd_series):

    tagging_dict = {}
    for element_index in range(len(pd_series.categories)): 
        entity_aspect = pd_series.categories[element_index]
        polarity = pd_series.polarities[element_index]
        start_index = [i for i, tpl in enumerate(pd_series.offsets) if int(pd_series['from'][element_index]) in tpl][0]
        end_index = [i for i, tpl in enumerate(pd_series.offsets) if int(pd_series['to'][element_index]) in tpl][0]
        if start_index == 0 and end_index != 0: 
            start_index = 1
        if (start_index, end_index) not in tagging_dict:
            tagging_dict[(start_index, end_index)] = [(entity_aspect, polarity)]
        else: 
            tagging_dict[(start_index, end_index)].append((entity_aspect, polarity))
            
    return tagging_dict

def split_entity_aspect_for_tripletextraction(train_df,EIDdata=False):
    Ent,Asp=[],[]
    for row in train_df.index:
        EA=list(train_df['entity_aspects'][int(row)]) 
        entity,aspect=[],[]
        s=len(EA) 
        for i in range(s):
            ea=EA[i]
            if ea!='None#':
                if EIDdata==True:
                    entity.append(ea.split('_')[0])
                    aspect.append(ea.split('_')[1])
                else:
                    entity.append(ea.split('#')[0])
                    aspect.append(ea.split('#')[1])
            else:
                entity.append('None#')
                aspect.append('None#')
        Ent.append(entity)
        Asp.append(aspect)
    train_df['entity']=Ent
    train_df['aspect']=Asp
    new_df=train_df[['tokens', 'entity','aspect','sentiment','offsets','Texts']]
    return new_df

def make_unique_entity(train_df,test_df):
    collect_entities = []
    for idx, row in enumerate(train_df.entity.values):
        collect_entities.extend(make_str_to_list(row))
    for idx, row in enumerate(test_df.entity.values):
        collect_entities.extend(make_str_to_list(row))
    unique_entity_aspects = set(collect_entities)
    
    # make NONE# comes first, so it labels as '0'
    unique_entity_aspects.remove('None#')
    rearrange_unique_entity_aspects = ['None#']

    rearrange_unique_entity_aspects.extend(unique_entity_aspects)
    label_dict = dict(zip(rearrange_unique_entity_aspects,range(len(rearrange_unique_entity_aspects))))
    # print('ENTITY LABEL LIST:\t',label_dict,'\n')
    return label_dict
def make_unique_aspect(train_df,test_df):
    collect_aspects = []
    for idx, row in enumerate(train_df.aspect.values):
        collect_aspects.extend(make_str_to_list(row))
    for idx, row in enumerate(test_df.aspect.values):
        collect_aspects.extend(make_str_to_list(row))
    unique_entity_aspects = set(collect_aspects)
    
    # make NONE# comes first, so it labels as '0'
    unique_entity_aspects.remove('None#')
    rearrange_unique_entity_aspects = ['None#']

    rearrange_unique_entity_aspects.extend(unique_entity_aspects)
    label_dict = dict(zip(rearrange_unique_entity_aspects,range(len(rearrange_unique_entity_aspects))))
    # print('ASPECT LABEL LIST:\t',label_dict,'\n')
    return label_dict

def make_unique_sentiment(train_df,test_df,prefix_data=None):
    collect_sentiments = []
    for idx, row in enumerate(train_df.sentiment.values):
        collect_sentiments.extend(make_str_to_list(row))
    for idx, row in enumerate(test_df.sentiment.values):
        collect_sentiments.extend(make_str_to_list(row))
    unique_entity_aspects = set(collect_sentiments)
    
    # make NONE# comes first, so it labels as '0'
    unique_entity_aspects.remove('None#')
    rearrange_unique_entity_aspects = ['None#']

    rearrange_unique_entity_aspects.extend(unique_entity_aspects)
    if prefix_data=='EIDdata': label_dict = dict(zip(['None#','negative','positive'],range(3)))
    else: label_dict = dict(zip(['None#','neutral','negative','positive'],range(4)))
    # print('SENTIMENT LABEL LIST:\t',label_dict,'\n')
    return label_dict

def rearrange_prediction(probabilities,threshold,pred_labels):
    for row in probabilities:
        row = row.tolist()
        for e in row:
            max_value = max(e)
            # opt1: take the value > threshold as final prediction
            if max_value > threshold:   max_index = [e.index(max_value)]
            else:   max_index = [0]
            pred_labels.extend(max_index)
    return pred_labels

def reutn_weight_list(train_df, label_dict, is_entity=False, is_aspect=False, is_sentimnet=False): 

    count_dict = {}
    weight_list = []
    
    if is_entity: column = 'entity'
    elif is_aspect: column = 'aspect'
    elif is_sentimnet: column = 'sentiment'
    
    for idx, row in train_df.iterrows():
        for item in make_str_to_list(row[column]):
            if item not in count_dict.keys(): 
                count_dict[item] = 1
            else: 
                count_dict[item] += 1

    
    sum_dict = sum(count_dict.values())
    for key in label_dict.keys(): 
    # sklearn formula
        weight = sum_dict/(count_dict[key]*len(label_dict))
        weight_list.append(round(weight, 2))
    
    return weight_list, count_dict