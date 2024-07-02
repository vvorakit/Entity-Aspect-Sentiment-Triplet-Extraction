import time
import data_preprocessing as preprocessing
import torch
import pandas as pd
import torch.nn.functional as F

def train_one_epoch_triplet(model, tokenizer, optimizer, train_df, entity_dict, aspect_dict, sentiment_dict, mps_device, batch_size=1):
    
    running_loss_entity = 0
    running_loss_aspect = 0
    running_loss_sentiment = 0
    
    for idx, batch in enumerate(preprocessing.batch_generator(train_df.values.tolist(), batch_size=batch_size)):
        batch_df = pd.DataFrame(batch, columns =['tokens', 'entity','aspect','sentiment','offsets','Texts'])
        tokenized_text = tokenizer(batch_df.Texts.values.tolist(),padding=False,return_tensors='pt').to(mps_device)

        batch_entity_labels,batch_aspect_labels,batch_sentiment_labels = [],[],[]
        
        for row_entity,row_aspect,row_sentiment in zip(batch_df.entity,batch_df.aspect,batch_df.sentiment):
            temp_row_label = [entity_dict[i] for i in preprocessing.make_str_to_list(row_entity)]
            batch_entity_labels.append(temp_row_label)

            temp_row_label = [aspect_dict[i] for i in preprocessing.make_str_to_list(row_aspect)]
            batch_aspect_labels.append(temp_row_label)

            temp_row_label = [sentiment_dict[i] for i in preprocessing.make_str_to_list(row_sentiment)]
            batch_sentiment_labels.append(temp_row_label)

        # initialize calculated gradients (from prev step)
        optimizer.zero_grad()

        # # pull all tensor batches required for training
        input_ids = tokenized_text['input_ids'].to(mps_device)
        attention_mask = tokenized_text['attention_mask'].to(mps_device)
        entity_labels = torch.as_tensor(batch_entity_labels).to(mps_device)
        aspect_labels = torch.as_tensor(batch_aspect_labels).to(mps_device)
        sentiment_labels = torch.as_tensor(batch_sentiment_labels).to(mps_device)
        # process
        outputs = model(input_ids, attention_mask=attention_mask,entity_labels=entity_labels,aspect_labels=aspect_labels,sentiment_labels=sentiment_labels)
        # extract loss
        loss = outputs.loss[0]
        running_loss_entity += outputs.loss[1]
        running_loss_aspect += outputs.loss[2]
        running_loss_sentiment += outputs.loss[3]
        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        optimizer.step() 
        
    avg_loss_entity = running_loss_entity/idx
    avg_loss_aspect = running_loss_aspect/idx
    avg_loss_sentiment = running_loss_sentiment/idx
    
    return avg_loss_entity, avg_loss_aspect, avg_loss_sentiment

def validate_one_epoch_triplet(model, tokenizer, val_df, entity_dict, aspect_dict, sentiment_dict, mps_device, batch_size=1):
    
    running_loss_entity = 0
    running_loss_aspect = 0
    running_loss_sentiment = 0
    
    for idx, batch in enumerate(preprocessing.batch_generator(val_df.values.tolist(), batch_size=batch_size)):
        batch_df = pd.DataFrame(batch, columns =['tokens', 'entity','aspect','sentiment','offsets','Texts'])
        tokenized_text = tokenizer(batch_df.Texts.values.tolist(),padding=False,return_tensors='pt').to(mps_device)

        batch_entity_labels,batch_aspect_labels,batch_sentiment_labels = [],[],[]
        
        for row_entity,row_aspect,row_sentiment in zip(batch_df.entity,batch_df.aspect,batch_df.sentiment):
            temp_row_label = [entity_dict[i] for i in preprocessing.make_str_to_list(row_entity)]
            batch_entity_labels.append(temp_row_label)

            temp_row_label = [aspect_dict[i] for i in preprocessing.make_str_to_list(row_aspect)]
            batch_aspect_labels.append(temp_row_label)

            temp_row_label = [sentiment_dict[i] for i in preprocessing.make_str_to_list(row_sentiment)]
            batch_sentiment_labels.append(temp_row_label)

        # # pull all tensor batches required for training
        input_ids = tokenized_text['input_ids'].to(mps_device)
        attention_mask = tokenized_text['attention_mask'].to(mps_device)
        entity_labels = torch.as_tensor(batch_entity_labels).to(mps_device)
        aspect_labels = torch.as_tensor(batch_aspect_labels).to(mps_device)
        sentiment_labels = torch.as_tensor(batch_sentiment_labels).to(mps_device)
        # process
        outputs = model(input_ids, attention_mask=attention_mask,entity_labels=entity_labels,aspect_labels=aspect_labels,sentiment_labels=sentiment_labels)
        # extract loss
        loss = outputs.loss[0]
        running_loss_entity += outputs.loss[1]
        running_loss_aspect += outputs.loss[2]
        running_loss_sentiment += outputs.loss[3]
        
    avg_loss_entity = running_loss_entity/idx
    avg_loss_aspect = running_loss_aspect/idx
    avg_loss_sentiment = running_loss_sentiment/idx
    
    return avg_loss_entity, avg_loss_aspect, avg_loss_sentiment

def obtain_results_sentiment(model, tokenizer, test_df, entity_dict, aspect_dict, sentiment_dict, path, inverted_lables_entity, inverted_lables_aspect, inverted_lables_sentiment, mps_device, batch_size=1): 
    named_entity_labels = list(entity_dict.keys())
    named_aspect_labels = list(aspect_dict.keys())
    named_sentiment_labels = list(sentiment_dict.keys())

    model.eval()
    results_entity, results_aspect, results_sentiment = pd.Series(), pd.Series(), pd.Series()

    
    for batch in preprocessing.batch_generator(test_df.values.tolist(), batch_size=batch_size):
        true_entity_labels, true_aspect_labels, true_sentiment_labels = [],[],[]
        pred_entity_labels, pred_aspect_labels, pred_sentiment_labels = [],[],[]
        batch_df = pd.DataFrame(batch, columns =['tokens', 'entity','aspect','sentiment','offsets','Texts'])
        tokenized_text = tokenizer(batch_df.Texts.values.tolist(),padding=False,return_tensors='pt').to(mps_device)
        for row_entity,row_aspect,row_sentiment in zip(batch_df.entity,batch_df.aspect,batch_df.sentiment):
            temp_row_label = [entity_dict[i] for i in preprocessing.make_str_to_list(row_entity)]
            true_entity_labels.extend(temp_row_label)

            temp_row_label = [aspect_dict[i] for i in preprocessing.make_str_to_list(row_aspect)]
            true_aspect_labels.extend(temp_row_label)

            temp_row_label = [sentiment_dict[i] for i in preprocessing.make_str_to_list(row_sentiment)]
            true_sentiment_labels.extend(temp_row_label)

        # pull all tensor batches required for testing
        input_ids = tokenized_text['input_ids'].to(mps_device)
        attention_mask = tokenized_text['attention_mask'].to(mps_device)

        # process
        outputs = model(input_ids, attention_mask=attention_mask)
        entity_logits = outputs.logits[0]
        aspect_logits = outputs.logits[1]
        sentiment_logits = outputs.logits[2]

        # get predicted probabilities, then find max as prediction
        entity_probabilities = F.softmax(entity_logits,dim=2)
        aspect_probabilities = F.softmax(aspect_logits,dim=2)
        sentiment_probabilities = F.softmax(sentiment_logits,dim=2)

        pred_entity_labels = preprocessing.rearrange_prediction(entity_probabilities,0.5,pred_entity_labels)
        pred_aspect_labels = preprocessing.rearrange_prediction(aspect_probabilities,0.5,pred_aspect_labels)
        pred_sentiment_labels = preprocessing.rearrange_prediction(sentiment_probabilities,0.5,pred_sentiment_labels)
        pred_entity = [inverted_lables_entity[i] for i in pred_entity_labels]
        pred_aspect = [inverted_lables_aspect[i] for i in pred_aspect_labels]
        pred_sentiment = [inverted_lables_sentiment[i] for i in pred_sentiment_labels]
        results_entity[len(results_entity)] = pred_entity
        results_aspect[len(results_aspect)] = pred_aspect
        results_sentiment[len(results_sentiment)] = pred_sentiment
    
    return results_entity, results_aspect, results_sentiment