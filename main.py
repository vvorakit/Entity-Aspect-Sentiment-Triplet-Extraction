import torch
import utils.data_preprocessing as preprocessing
from EASTE_transformers.src.transformers.models.bert import BertTokenClassificationForTripletExtraction
from EASTE_transformers.src.transformers import AutoTokenizer
import utils.training_utils as train 
import time
from sklearn.model_selection import train_test_split
from torch.optim import AdamW, SGD

EPOCHS = 10

if __name__ == '__main__': 

    pretrained_tokenizer_model = 'bert' # this code is for bert only

    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print (x)
    else:
        print ("MPS device not found.")
    torch.cuda.empty_cache()

    train_path ='data/...' # add path to raw training SemEval16 dataset
    test_path ='data/...' # add path to raw test SemEval16 dataset
    
    if train_path and test_path:
        train_df = preprocessing.xml_to_dataframe(train_path)
        test_df = preprocessing.xml_to_dataframe(test_path)
    else: 
        print('Datasets not found')

    # ### pre-process datasets 
    train_df = preprocessing.get_semeval2016task5(train_df,model=pretrained_tokenizer_model)
    test_df = preprocessing.get_semeval2016task5(test_df,model=pretrained_tokenizer_model)
    train_df = preprocessing.get_semeval2016task5_multi_annotation(train_df)
    test_df = preprocessing.get_semeval2016task5_multi_annotation(test_df)

    entity_dict = {'None#': 0, 
               'RESTAURANT': 1, 
               'DRINKS': 2,
               'FOOD': 3,
               'SERVICE': 4,
               'AMBIENCE': 5,
               'LOCATION': 6}
    aspect_dict = {'None#': 0,
                'MISCELLANEOUS': 1,
                'QUALITY': 2,
                'GENERAL': 3,
                'PRICES': 4,
                'STYLE_OPTIONS': 5}
    sentiment_dict = {'None#': 0, 
                    'neutral': 1, 
                    'negative': 2, 
                    'positive': 3}

    inverted_lables_entity = {v:k for (k, v) in entity_dict.items()}
    inverted_lables_aspect = {v:k for (k, v) in aspect_dict.items()}
    inverted_lables_sentiment = {v:k for (k, v) in sentiment_dict.items()}
    
    weights_list_entity, count_entity = preprocessing.reutn_weight_list(train_df, entity_dict, is_entity=True)
    weights_list_aspect, count_aspect = preprocessing.reutn_weight_list(train_df, aspect_dict, is_aspect=True)
    weights_list_sentiment, count_sentiment = preprocessing.reutn_weight_list(train_df, sentiment_dict, is_sentimnet=True)

    if mps_device: 
        weights_entity = torch.tensor(weights_list_entity).to(mps_device)
        weights_aspect = torch.tensor(weights_list_aspect).to(mps_device)
        weights_sentiment = torch.tensor(weights_list_sentiment).to(mps_device)

    model_path = "google-bert/bert-base-uncased"
    model = BertTokenClassificationForTripletExtraction.from_pretrained(model_path,num_entity_labels=len(entity_dict),num_aspect_labels=len(aspect_dict),num_sentiment_labels=len(sentiment_dict), weights_entity=weights_entity, weights_aspect=weights_aspect, weights_sentiment=weights_sentiment)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    if mps_device:
        model.to(mps_device)


    test_df, val_df = train_test_split(test_df, test_size=0.15)
    label_dict = {'None#': 0, 
              'SERVICE#GENERAL': 1, 
              'FOOD#STYLE_OPTIONS': 2, 
              'DRINKS#STYLE_OPTIONS': 3, 
              'DRINKS#QUALITY': 4, 
              'DRINKS#PRICES': 5, 
              'FOOD#PRICES': 6, 
              'RESTAURANT#GENERAL': 7, 
              'LOCATION#GENERAL': 8, 
              'AMBIENCE#GENERAL': 9,
              'RESTAURANT#PRICES': 10, 
              'RESTAURANT#MISCELLANEOUS': 11, 
              'FOOD#QUALITY': 12}
     
    optimizer = SGD(model.parameters(), 1e-3)
    batches = preprocessing.batch_generator(train_df.values.tolist(), batch_size=1)
    total_batches = len(list(batches))
    loss_dict = {'train_loss_entity': [],
                'train_loss_aspect': [],
                'train_loss_sentiment': [],
                'val_loss_entity': [],
                'val_loss_aspect': [],
                'val_loss_sentiment': []}

    for epoch in range(1, EPOCHS): 
        
        start = time.time()
        print('EPOCH {}:'.format(epoch))
        model.train(True)
        train_loss = train.train_one_epoch(model, tokenizer, train_df, label_dict)
        loss_dict['train_loss'].append(train_loss)
        end = time.time()

        model.train(False)
        validation_loss = train.validate_one_epoch(model, tokenizer, val_df, label_dict)
        loss_dict['val_loss'].append(validation_loss)
        
        print('Training LOSS: {}, Validation LOSS: {}\t Training TIME: {:.2f}secs'.format(train_loss, validation_loss, end-start))

    

