import os
from torch.nn.utils.clip_grad import clip_grad_norm
import torch
import torch.nn as nn
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModel, AdamW, AutoModelForSequenceClassification, \
    get_linear_schedule_with_warmup, BertTokenizer, BertModel
from tqdm import tqdm
from transformers.adapters import RobertaAdapterModel


# implementation of FL_pooler class
class FL_pooler(nn.Module):
    def __init__(self, drop_rate=0.2):
        super(FL_pooler, self).__init__()
        D_in, D_out = 3, 1
        D_pooler_out = 768
        # get file path of fine-tuned regressor
        p_V_path, p_A_path, p_D_path = "", "", ""
        self.v_bert = RobertaAdapterModel.from_pretrained("vinai/bertweet-base")
        self.a_bert = RobertaAdapterModel.from_pretrained("vinai/bertweet-base")
        self.d_bert = RobertaAdapterModel.from_pretrained("vinai/bertweet-base")
        self.v_bert.load_state_dict(torch.load(p_V_path), strict=False)
        self.a_bert.load_state_dict(torch.load(p_A_path), strict=False)
        self.d_bert.load_state_dict(torch.load(p_D_path), strict=False)
        self.pooler = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(D_in, D_out))
        self.clf = nn.Linear(D_pooler_out, 3)

    def forward(self, input_ids, attention_masks):
        # add adapters and activate them
        self.v_bert.add_adapter("v_adapter")
        self.a_bert.add_adapter("a_adapter")
        self.d_bert.add_adapter("d_adapter")
        # Activate the adapter
        self.v_bert.train_adapter("v_adapter")
        self.a_bert.train_adapter("a_adapter")
        self.d_bert.train_adapter("d_adapter")
        # get outputs for three models
        v_output = self.v_bert(input_ids, attention_masks)[1]
        a_output = self.a_bert(input_ids, attention_masks)[1]
        d_output = self.d_bert(input_ids, attention_masks)[1]
        # concat the output and align the dimensions
        concated = torch.cat((v_output, a_output, d_output), dim=1)
        concated = torch.reshape(concated, (concated.size()[0], 3, concated.size()[1] // 3))
        concated = concated.transpose(-2, -1)
        # for debugging
        print(concated.size())
        output = self.pooler(concated.squeeze())
        # get logits
        logits = self.clf(output)
        return logits


# evaluate model performance (acc)
def evaluate(model, test_data: Dataset, batch_size: int = 64):
    model.eval()
    with torch.no_grad():
        total_correct = 0
        losses = []
        loss_function = nn.CrossEntropyLoss()
        for i in tqdm(range(0, len(test_data), batch_size)):
            batch = test_data[i:i + batch_size]
            input_ids, attention_mask = torch.tensor(batch["input_ids"]).to(device), \
                                        torch.tensor(batch["attention_mask"]).to(device)
            # get classifier logit scores
            outputs = model(input_ids, attention_masks=attention_mask)
            # ground truth labels
            test_labels = torch.tensor(batch["labels"]).to(device)
            probs = nn.functional.softmax(outputs)
            # get model output labels
            labels = torch.argmax(probs, dim=1)
            # calculate cross entropy loss
            loss = loss_function(probs, batch["labels"])
            losses.append(loss)
            # calculate accuracy
            correct = torch.tensor(labels == test_labels)
            total_correct += sum(correct)
            acc = total_correct / len(test_data)
        return acc, torch.mean(torch.tensor(losses))


# trainer
def train(model, train_data: Dataset, val_data: Dataset,
          batch_size: int = 64, max_epochs: int = 10,
          file_path: str = "checkpoints/pooler_clf", clip_value: int = 2):
    lr, lr_mul = 5e-5, 1
    # weight_decay = 1e-5
    eps = 1e-8
    # initialize optimizer
    adam = AdamW(model.parameters(),
                 # [{'params': bert_param},
                 # {'params': reg_param, 'lr': lr * lr_mul, 'weight_decay': weight_decay}],
                 lr=lr,
                 eps=eps,
                 # weight_decay=weight_decay,
                 )
    # initialize scheduler
    total_steps = len(train_data) * max_epochs
    scheduler = get_linear_schedule_with_warmup(adam,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    loss_function = nn.CrossEntropyLoss()
    # store historical accs
    val_accs = []
    for epoch in range(max_epochs):
        print("Epoch {} of {}".format(epoch + 1, max_epochs))
        # Training code
        print("Training...")
        model.train()
        for i in tqdm(range(0, len(train_data), batch_size)):
            batch = train_data[i:i + batch_size]
            # calculate loss and do SGD
            input_ids, attention_mask = torch.tensor(batch["input_ids"]).to(device), torch.tensor(
                batch["attention_mask"]).to(device)
            batch_labels = torch.tensor(batch["labels"]).to(device)
            logits = model(input_ids, attention_mask)
            # loss = logits[0]
            loss = loss_function(logits, batch_labels)
            adam.zero_grad()
            loss.backward()
            # prevent gradient vanishing
            clip_grad_norm(model.parameters(), clip_value)
            adam.step()
            scheduler.step()
        # Test on validation data
        print("Evaluating on validation data...")
        val_acc, loss = evaluate(model, val_data)
        print("Validation acc: {:.3f}, cross entropy loss: {:.3f}".format(val_acc, loss))
        val_accs.append(val_acc)
        break
    #         torch.save(BertweetRegressor.bertweet.state_dict(),
    #                    "{}/epoch{}@sid{}.pt".format(file_path, epoch, os.environ['SLURM_JOB_ID']))
    r_scores = torch.tensor(val_accs)
    print("Best val achieved at epoch {}, with r2 score {}, slurm_job_id: {}".format(torch.argmax(r_scores),
                                                                                     torch.max(r_scores),
                                                                                     os.environ['SLURM_JOB_ID']))


def preprocess_data(dataset, tokenizer):
    dataset = dataset.map(lambda x: tokenizer(x['text'],
                                              add_special_tokens=True,
                                              padding="max_length",
                                              max_length=128,
                                              truncation="longest_first",
                                              # return_tensors="pt",
                                              # return_attentiton_mask=True,
                                              ))
    return dataset

# main training script:
if __name__ == '__main__':
    model_name = "vinai/bertweet-base"
    # load and preprocess dataset
    clf_dataset = load_dataset("csv", data_files={"train": "sar_and_meta_train.csv", "test": "sar_and_meta_test.csv"})
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    clf_dataset["train"] = preprocess_data(clf_dataset["train"], tokenizer)
    # split training set into traindev
    val_size = 0.1
    seed = 42
    # split data
    clf_split = clf_dataset["train"].train_test_split(val_size, seed=seed)
    clf_dataset["train"] = clf_split["train"]
    clf_dataset["val"] = clf_split["test"]

    # initialize regressor model
    model = FL_pooler()
    # assign computing resources
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU.")
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
    model.to(device)

    # train the regressor
    train(model, train_data=clf_dataset["train"], val_data=clf_dataset["val"])