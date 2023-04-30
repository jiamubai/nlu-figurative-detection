import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import transformers
# import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup, BertTokenizer, BertModel
from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from torch.nn.utils import clip_grad_norm
import gc

class BertweetRegressor(nn.Module):
    
    def __init__(self, drop_rate=0.2, freeze_bertweet=False):
        
        super(BertweetRegressor, self).__init__()
        D_in, D_out = 768, 3
        
        self.bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
        model_p = torch.load('checkpoints/multi_reg/epoch2@sid129547.pt')
        self.bertweet.load_state_dict(model_p)
        
        #for param in self.bertweet.parameters():
        #    param.requires_grad = False
        
        self.regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(D_in, D_out))
        self.double()
        
    def forward(self, input_ids, attention_masks):
        
        outputs = self.bertweet(input_ids, attention_masks)
        class_label_output = outputs[1]
        outputs = self.regressor(class_label_output)
        return outputs
    
def evaluate_classify(model, loss_function, test_dataloader, device):
    test_loss = []
    correct = 0
    for batch in test_dataloader:
        batch_inputs, batch_masks, batch_labels = \
                                 tuple(b.to(device) for b in batch)
        with torch.no_grad():
            outputs = model(batch_inputs, batch_masks)

            loss = loss_function(outputs, batch_labels)
            test_loss.append(loss)
            pred = outputs.data.max(1, keepdim=True)[1]
            correct += pred.eq(batch_labels.data.view_as(pred)).sum()
    
    test_loss = np.array(test_loss)
    return np.sum(test_loss)/len(test_loss), correct/len(test_dataloader.dataset)

def train_with_vad(model, optimizer, scheduler, loss_function, epochs,       
          train_dataloader, val_dataloader, device, clip_value=2):

    train_loss_list = []
    val_loss_list = []

    for epoch in range(epochs):
        print(epoch)
        print("-----")
        #best_loss = 1e10
        model.train()
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(num_trainable_params)
        train_loss = []
        correct = 0
        for step, batch in enumerate(train_dataloader): 
            batch_inputs, batch_masks, batch_labels = \
                               tuple(b.to(device) for b in batch)
            output = model(batch_inputs, batch_masks)
            loss = loss_function(output, 
                             batch_labels)
            train_loss.append(loss.data)
            
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(batch_labels.view_as(pred)).sum().item()
            optimizer.zero_grad()
            loss.backward()
#             clip_grad_norm(model.parameters(), clip_value)
            optimizer.step()
#             scheduler.step()


        train_loss = [loss.cpu().item() for loss in train_loss]
        train_loss = np.sum(train_loss) / len(train_loss)
        train_acc = correct/ len(train_dataloader.dataset)
        
        val_loss, val_acc = evaluate_classify(model, loss_function, val_dataloader, device)

        print('train loss: ', train_loss, train_acc)
        print('val loss: ', val_loss, val_acc)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss.cpu().item())     

    return model, train_loss_list, val_loss_list

def create_dataloaders(inputs, masks, labels, batch_size):
    input_tensor = torch.tensor(inputs)
    mask_tensor = torch.tensor(masks)
    labels_tensor = torch.tensor(labels)
    dataset = TensorDataset(input_tensor, mask_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=True)
    return dataloader
  
# load dataset
train_fl = pd.read_csv("train_with_vua.csv")[:10000]
test_fl = pd.read_csv("test_with_vua.csv")

val_size = 0.1
seed = 42
batch_size = 32
pretrain_vad = True # change this based on pretrained model choice
epochs = 10
lr = 5e-5
eps=1e-8
times = 1000
weight_decay = 1e-7
D_out = 3 # change this based on label numbers
loss_function = nn.CrossEntropyLoss()

# load tokenizer 
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
    
encoded = tokenizer(text=train_fl.text.tolist(), 
            add_special_tokens=True,
            padding='max_length',
            truncation='longest_first',
            max_length=128,
            return_attention_mask=True)

# split train/val/test and create dataloaders
input_ids = encoded['input_ids']
attention_mask = encoded['attention_mask']

labels = train_fl.label.to_numpy()

train_inputs, val_inputs, train_labels, val_labels = \
            train_test_split(input_ids, labels, test_size=val_size, random_state=seed)
train_masks, val_masks, _, _= train_test_split(attention_mask, labels, test_size=val_size, random_state=seed)

train_dataloader = create_dataloaders(train_inputs, train_masks, train_labels, batch_size)
val_dataloader = create_dataloaders(val_inputs, val_masks, val_labels, batch_size)

# clean cuda in case full
torch.cuda.empty_cache()
gc.collect()

# load pretrained model
model = BertweetRegressor(drop_rate=0.0)
"""
if pretrain_vad:
    model = torch.load('checkpoints/epoch4@sid32907345.pt') 

    drop_rate = 0.0
    D_in = 768
    D_out = D_out # change this based on label numbers
    model.regressor = nn.Sequential(
                          nn.Dropout(drop_rate),
                          nn.Linear(D_in, D_out))
    model.double()
"""

# connect to cuda
if torch.cuda.is_available():       
    device = torch.device("cuda")
    print("Using GPU.")
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")
model.to(device)

# finetune pretrained parameters, hypertune new layer
params_1x = [param for name, param in model.named_parameters() if 'regressor' not in str(name)]
optimizer = AdamW([{'params':params_1x}, {'params': model.regressor.parameters(), 'lr': lr*times, 'weight_decay': 0} ],
                  lr=lr,
                  eps=eps, weight_decay = weight_decay)

total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,       
                 num_warmup_steps=0, num_training_steps=total_steps)

model_meta, train_loss_list, val_loss_list = train_with_vad(model, optimizer, scheduler, loss_function, epochs, 
                                                        train_dataloader, val_dataloader, device, clip_value=2)
