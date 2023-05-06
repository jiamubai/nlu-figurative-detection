import numpy as np
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
        print(outputs[1].size())
        class_label_output = outputs[1]
        
        if res == 'clf':
            final_outputs = self.clf(class_label_output)
        elif res == 'res':
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
def evaluate(model, test_dataloader):
    for batch in test_dataloader:
        reg_input_ids, reg_attention_mask, reg_labels = torch.tensor(batch["input_ids"]).to(device), torch.tensor(batch["attention_mask"]).to(device)
        reg_labels = torch.tensor(np.array([batch["V"], batch["A"], batch["D"]]).T).float().to(device)
        
        reg_output = model(reg_input_ids, reg_attention_mask, 'reg')
        
        loss_function = nn.MSELoss(reduction="sum")
        loss = loss_function(reg_output, reg_labels)
        r2_score = cal_r2_score(reg_output, reg_labels)
        return loss, r2_score
    
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
    
    test_loss = [loss.cpu().item() for loss in test_loss]
    return np.sum(test_loss)/len(test_loss), correct/len(test_dataloader.dataset)

def get_dataloader(clf_loader, reg_loader):
    if random.random() > 0:
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
    lr, lr_mul =5e-5, 1
    weight_decay = 5e-5
    eps = 1e-8
    adam = AdamW(BertweetMulti.parameters(), lr=lr)
    loss_function_reg = nn.MSELoss(reduction="sum")
    loss_function_clf = nn.CrossEntropyLoss()
    
    # store historical residuals
    r_scores = []
    for epoch in range(max_epochs):
        print("Epoch {} of {}".format(epoch + 1, max_epochs))

        # Training code
        print("Training...")
        BertweetMulti.train()
        
        train_loss = []
        for j in range(200):
            dataloader_train, res = get_dataloader(clf_train_dataloader, reg_train_dataloader)
            batch = next(iter(dataloader_train))
            
            if res == 'clf':
                print(len(batch["input_ids"]))
                clf_input_ids, clf_attention_mask, clf_labels = torch.stack(tuple(batch["input_ids"])).to(device), torch.stack(tuple(batch["attention_mask"])).to(device), torch.stack(tuple(batch["label"])).to(device)

                clf_output = BertweetMulti(clf_input_ids, clf_attention_mask, res)
                clf_loss = loss_function_clf(clf_output, clf_labels)
                train_loss.append(clf_loss.data)
                
                adam.zero_grad()
                clf_loss.backward()
                adam.step()
                
            elif res == 'reg':
                reg_input_ids, reg_attention_mask = torch.tensor(batch["input_ids"]).to(device), torch.tensor(batch["attention_mask"]).to(device)
                reg_labels = torch.cat([batch["V"].unsqueeze(1), batch["A"].unsqueeze(1), batch["D"].unsqueeze(1)], dim=1).float().to(device)
                
                reg_output = BertweetMulti(reg_input_ids, reg_attention_mask, res)
                reg_loss = loss_function_reg(reg_output, reg_labels)
                train_loss.append(reg_loss.data)
                
                adam.zero_grad()
                reg_loss.backward()
                adam.step()
            
        # Test on validation data
        print("Evaluating on validation data...")
        reg_val_loss, reg_r2 = evaluate(BertweetMulti, reg_val_dataloader)
        
        clf_val_loss, clf_val_acc = evaluate_classify(BertweetMulti, loss_function_clf , clf_val_dataloader, device)
        
        train_loss = [loss.cpu().item() for loss in train_loss]
        train_loss = np.sum(train_loss) / len(train_loss)
        #train_acc = correct/ len(train_dataloader.dataset)
        
        print('train loss: ', train_loss)
        print('reg_val loss: ', reg_val_loss, reg_r2)
        print('clf_val loss: ', clf_val_loss, clf_val_acc)
        
        
        torch.save(BertweetMulti.bertweet.state_dict(), "{}/epoch{}@sid{}.pt".format(file_path, epoch, os.environ['SLURM_JOB_ID']))
#     print(r_scores)
    r_scores = torch.tensor(r_scores)
    print("Best val achieved at epoch {}, with r2 score {}, slurm_job_id: {}".format(torch.argmax(r_scores), torch.max(r_scores), os.environ['SLURM_JOB_ID']))


def preprocess_data(dataset, tokenizer):
    dataset = dataset.map(lambda x: tokenizer(x['text'],
                                    add_special_tokens=True,
                                    padding="max_length",
                                    max_length=128,
                                    truncation="longest_first",
                                    #return_attentiton_mask=True,
                                    ))
    return dataset


if __name__ == '__main__':
    # main training script:
    model_name = "vinai/bertweet-base"
    # load and preprocess dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    reg_dataset = load_dataset("csv", data_files={"train": "norm_emobank_train.csv", "test": "norm_emobank_test.csv"})
    clf_dataset = load_dataset("csv", data_files={"train": "train_with_vua.csv", "test": "test_with_vua.csv"})
    
    reg_dataset["train"] = preprocess_data(reg_dataset["train"], tokenizer)

    clf_dataset["train"] = preprocess_data(clf_dataset["train"], tokenizer)

    # split training set into traindev
    val_size = 0.1
    seed = 42
    # regression data
    reg_split = reg_dataset["train"].train_test_split(val_size, seed=seed)
    reg_dataset["train"] = reg_split["train"]
    reg_dataset["val"] = reg_split["test"]
    # classification data
    clf_split = clf_dataset["train"].train_test_split(val_size, seed=seed)
    clf_dataset["train"] = clf_split["train"]
    clf_dataset["val"] = clf_split["test"]

    batch_size = 128

    # Create dataloaders for regression dataset
    reg_train_loader = DataLoader(reg_dataset["train"], batch_size=batch_size, shuffle=True)
    reg_val_loader = DataLoader(reg_dataset["val"], batch_size=batch_size)

    # Create dataloaders for classification dataset
    clf_train_loader = DataLoader(clf_dataset["train"], batch_size=batch_size, shuffle=True)
    clf_val_loader = DataLoader(clf_dataset["val"], batch_size=batch_size)
    
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
