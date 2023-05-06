import numpy as np
import pandas as pd
import os
# from sklearn.metrics import r2_score
import torch
import torch.nn as nn
from datasets import load_dataset, Dataset
import transformers
from transformers import AutoTokenizer, AutoModel, AdamW, AutoModelForSequenceClassification, \
    get_linear_schedule_with_warmup, BertTokenizer, BertModel
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Bertweet regressor
class BertweetMulti(nn.Module):

    def __init__(self, drop_rate=0.2, freeze_bertweet=False):
        super(BertweetMulti, self).__init__()
        D_in, D_out = 768, 3

        self.bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
        
        self.regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(D_in, D_out))
        
        self.clf = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(D_in, D_out))
        self.double()

    def forward(self, input_ids, attention_masks, res):
        outputs = self.bertweet(input_ids, attention_masks)
        class_label_output = outputs[1]
        
        if res == 'clf':
            final_outputs = self.clf(class_label_output)
            return final_outputs
        elif res == 'reg':
            final_outputs = self.regressor(class_label_output)
            return final_outputs
    
# calculate residual
def cal_r2_score(outputs, labels):
    labels_mean = torch.mean(labels, dim=0)
#     outputs = torch.sum(outputs, dim=1)
#     labels = torch.sum(labels, dim=1)
#     labels_mean = torch.mean(labels)
    ss_tot = torch.sum((labels - labels_mean) ** 2, dim=0)
    ss_res = torch.sum((labels - outputs) ** 2, dim=0)
    r2 = 1 - ss_res / ss_tot
    return torch.mean(r2)

# evaluate model performace (R2 score)
def evaluate_reg(model, test_dataloader):
    for batch in test_dataloader:
        reg_input_ids, reg_attention_mask, reg_labels = tuple(b.to(device) for b in batch)
        
        reg_output = model(reg_input_ids, reg_attention_mask, 'reg')
        
        loss_function = nn.MSELoss(reduction="sum")
        loss = loss_function(reg_output, reg_labels)
        r2_score = cal_r2_score(reg_output, reg_labels)
        return loss, r2_score
    
def evaluate_classify(model, loss_function, test_dataloader, device):
    test_loss = []
    correct = 0
    
    all_labels = []
    all_preds = []
    
    for batch in test_dataloader:
        batch_inputs, batch_masks, batch_labels = \
                                 tuple(b.to(device) for b in batch)
        with torch.no_grad():
            outputs = model(batch_inputs, batch_masks, 'clf')

            loss = loss_function(outputs, batch_labels)
            test_loss.append(loss)
            pred = outputs.data.max(1, keepdim=True)[1]
            correct += pred.eq(batch_labels.data.view_as(pred)).sum()
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(batch_labels.tolist())
            
    f1 = f1_score(all_labels, all_preds, average='weighted')
    test_loss = [loss.cpu().item() for loss in test_loss]
    return np.sum(test_loss)/len(test_loss), correct/len(test_dataloader.dataset), f1

def get_dataloader(clf_loader, reg_loader):
    if random.random() > 0.5:
        res = 'clf'
        return clf_loader, res
    else:
        res = 'reg'
        return reg_loader, res

# trainer
def train(BertweetMulti, reg_train_dataloader, reg_val_dataloader, clf_train_dataloader, clf_val_dataloader,
          batch_size: int = 64, max_epochs: int = 15,
          file_path: str = "checkpoints/multi_reg"):
    
    # split the params of regressor
    lr =1e-4
    weight_decay = 5e-5
    eps = 1e-8
    adam = AdamW(BertweetMulti.parameters(), lr=lr, eps=eps)
    loss_function_reg = nn.MSELoss(reduction="sum")
    loss_function_clf = nn.CrossEntropyLoss()
    
    # store historical residuals
    for epoch in range(10):
        print("Epoch {} of {}".format(epoch + 1, max_epochs))

        # Training code
        print("Training...")
        BertweetMulti.train()
        
        train_loss = []
        for j in range(100):
            dataloader_train, res = get_dataloader(clf_train_dataloader, reg_train_dataloader)
            batch = next(iter(dataloader_train))
            
            if res == 'clf':
                clf_input_ids, clf_attention_mask, clf_labels = tuple(b.to(device) for b in batch)

                clf_output = BertweetMulti(clf_input_ids, clf_attention_mask, res)
                clf_loss = loss_function_clf(clf_output, clf_labels)
                train_loss.append(clf_loss.data)
                
                adam.zero_grad()
                clf_loss.backward()
                adam.step()
                
                print(res, clf_loss)
                
            elif res == 'reg':
                reg_input_ids, reg_attention_mask, reg_labels = tuple(b.to(device) for b in batch)
                
                reg_output = BertweetMulti(reg_input_ids, reg_attention_mask, res)

                reg_loss = loss_function_reg(reg_output, reg_labels)

                train_loss.append(reg_loss.data)
                
                adam.zero_grad()
                reg_loss.backward()
                adam.step()
                
                print(res, reg_loss)
            
        # Test on validation data
        print("Evaluating on validation data...")
        reg_val_loss, reg_r2 = evaluate_reg(BertweetMulti, reg_val_dataloader)
        
        clf_val_loss, clf_val_acc, clf_f1 = evaluate_classify(BertweetMulti, loss_function_clf , clf_val_dataloader, device)
        
        train_loss = [loss.cpu().item() for loss in train_loss]
        train_loss = np.sum(train_loss) / len(train_loss)
        #train_acc = correct/ len(train_dataloader.dataset)
        
        print('train loss: ', train_loss)
        print('reg_val loss: ', reg_val_loss, 'reg_val r2: ', reg_r2)
        print('clf_val loss: ', clf_val_loss, "clf_val acc: ", clf_val_acc,  "clf_val f1: ", clf_val_f1)
    
    
def preprocess_data(dataset, tokenizer):
    dataset = dataset.map(lambda x: tokenizer(x['text'],
                                    add_special_tokens=True,
                                    padding="max_length",
                                    max_length=128,
                                    truncation="longest_first",
                                    #return_attentiton_mask=True,
                                    ))
    return dataset


def create_dataloaders(inputs, masks, labels, batch_size):
    input_tensor = torch.tensor(inputs)
    mask_tensor = torch.tensor(masks)
    labels_tensor = torch.tensor(labels)
    dataset = TensorDataset(input_tensor, mask_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=True)
    return dataloader

def create_dataloader_clf(train, tokenizer):
    val_size=0.1
    seed = 42
    encoded = tokenizer(text=train.text.tolist(), 
                add_special_tokens=True,
                padding='max_length',
                truncation='longest_first',
                max_length=128,
                return_attention_mask=True)

    # split train/val/test and create dataloaders
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']

    labels = train.label.to_numpy()

    train_inputs, val_inputs, train_labels, val_labels = \
                train_test_split(input_ids, labels, test_size=val_size, random_state=seed)
    train_masks, val_masks, _, _= train_test_split(attention_mask, labels, test_size=val_size, random_state=seed)

    train_dataloader = create_dataloaders(train_inputs, train_masks, train_labels, batch_size)
    val_dataloader = create_dataloaders(val_inputs, val_masks, val_labels, batch_size)
    
    return train_dataloader, val_dataloader

def create_dataloader_reg(train, tokenizer):
    val_size=0.1
    seed = 42
    encoded = tokenizer(text=train.text.tolist(), 
                add_special_tokens=True,
                padding='max_length',
                truncation='longest_first',
                max_length=128,
                return_attention_mask=True)

    # split train/val/test and create dataloaders
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']

    V = train.V.to_numpy()
    A = train.A.to_numpy()
    D = train.D.to_numpy()

    train_inputs, val_inputs, train_V, val_V, train_A, val_A, train_D, val_D= \
                train_test_split(input_ids, V, A, D, test_size=val_size, random_state=seed)
    train_masks, val_masks, _, _, _, _, _, _= train_test_split(attention_mask, V, A, D, test_size=val_size, random_state=seed)
    
    train_labels = torch.tensor(np.array([train_V, train_A, train_D]).T).double()
    val_labels = torch.tensor(np.array([val_V, val_A, val_D]).T).double()

    train_dataloader = create_dataloaders(train_inputs, train_masks, train_labels, batch_size)
    val_dataloader = create_dataloaders(val_inputs, val_masks, val_labels, batch_size)
    
    return train_dataloader, val_dataloader




if __name__ == '__main__':
    # main training script:
    model_name = "vinai/bertweet-base"
    # load and preprocess dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    clf_train = pd.read_csv("sar_and_meta_train.csv")[:10000]
    clf_test = pd.read_csv("sar_and_meta_test.csv")
    reg_train = pd.read_csv("norm_emobank_train.csv")[:10000]
    reg_test = pd.read_csv("norm_emobank_test.csv")
    
    clf_train_loader, clf_val_loader = create_dataloader_clf(clf_train, tokenizer)
    reg_train_loader, reg_val_loader = create_dataloader_reg(reg_train, tokenizer)
    
    # initialize regressor model
    reg = BertweetMulti()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU.")
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
    reg.to(device)

    # train the regressor
    train(reg, reg_train_loader, reg_val_loader, clf_train_loader, clf_val_loader)
