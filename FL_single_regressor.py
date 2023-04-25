import os
from torch.nn.utils.clip_grad import clip_grad_norm
import torch
import torch.nn as nn
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModel, AdamW, AutoModelForSequenceClassification, \
    get_linear_schedule_with_warmup, BertTokenizer, BertModel
from tqdm import tqdm


# Bertweet regressor
class BertweetRegressor(nn.Module):
    def __init__(self, drop_rate=0.2, freeze_bertweet=False):
        super(BertweetRegressor, self).__init__()
        D_in, D_out = 768, 1

        self.bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
        self.regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(D_in, D_out))

    def forward(self, input_ids, attention_masks):
        outputs = self.bertweet(input_ids, attention_masks)
        class_label_output = outputs[1]
        outputs = self.regressor(class_label_output)
        return outputs


# calculate residual
def cal_r2_score(outputs, labels):
    outputs = outputs.squeeze()
    assert outputs.size() == labels.size()
    labels_mean = torch.mean(labels)
    ss_tot = torch.sum((labels - labels_mean) ** 2)
    ss_res = torch.sum((labels - outputs) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


# evaluate model performace (R2 score)
def evaluate(model, test_data: Dataset):
    model.eval()
    with torch.no_grad():
        input_ids, attention_mask = torch.tensor(test_data["input_ids"]).to(device), \
                                    torch.tensor(test_data["attention_mask"]).to(device)
        test_labels = torch.tensor(test_data["A"]).float().to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=test_labels)

        # loss_function = nn.MSELoss()
        # loss = loss_function(outputs.squeeze(), test_labels)
        # print(outputs)
#         print(outputs[1].squeeze()[:100], test_labels[:100])
        r2_score = cal_r2_score(outputs[1], test_labels)
        return outputs[0], r2_score
        # return 0, 0


# trainer
def train(BertweetRegressor, train_data: Dataset, val_data: Dataset,
          batch_size: int = 64, max_epochs: int = 10,
          file_path: str = "checkpoints/single_reg", clip_value: int = 2):
    lr, lr_mul = 5e-6, 1
    weight_decay = 1e-7
    eps = 1e-6

    # initialize optimizer
    adam = AdamW(BertweetRegressor.parameters(),
                 # [{'params': bert_param},
                 # {'params': reg_param, 'lr': lr * lr_mul, 'weight_decay': weight_decay}],
                 lr=lr,
                 eps=eps,
                 weight_decay=weight_decay,
                 )

    # initialize scheduler
    total_steps = len(train_data) * max_epochs
    scheduler = get_linear_schedule_with_warmup(adam,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    loss_function = nn.MSELoss()
    # store historical residuals
    r_scores = []
    for epoch in range(max_epochs):
        print("Epoch {} of {}".format(epoch + 1, max_epochs))

        # Training code
        print("Training...")
        BertweetRegressor.train()
        for i in tqdm(range(0, len(train_data), batch_size)):
            batch = train_data[i:i + batch_size]
            # calculate loss and do SGD
            input_ids, attention_mask = torch.tensor(batch["input_ids"]).to(device), torch.tensor(
                batch["attention_mask"]).to(device)
            batch_labels = torch.tensor(batch["A"]).float().to(device)
            logits = BertweetRegressor(input_ids, attention_mask=attention_mask, labels=batch_labels)
            loss = logits[0]
            # loss = loss_function(logits, batch_labels)
            adam.zero_grad()
            loss.backward()
            # prevent gradient vanishing
            clip_grad_norm(BertweetRegressor.parameters(), clip_value)
            adam.step()
            scheduler.step()

        # Test on validation data
        print("Evaluating on validation data...")
        val_loss, r2 = evaluate(BertweetRegressor, val_data)
        print("Validation loss: {:.3f}, r2 score: {}".format(val_loss, r2))
        r_scores.append(r2)
#         break
        torch.save(BertweetRegressor.state_dict(),
                   "{}/epoch{}@sid{}.pt".format(file_path, epoch, os.environ['SLURM_JOB_ID']))
    r_scores = torch.tensor(r_scores)
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


if __name__ == '__main__':
    # main training script:
    model_name = "vinai/bertweet-base"
    # load and preprocess dataset
    reg_dataset = load_dataset("csv", data_files={"train": "norm_emobank_train.csv", "test": "norm_emobank_test.csv"})
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    reg_dataset["train"] = preprocess_data(reg_dataset["train"], tokenizer)
    # split training set into traindev
    val_size = 0.1
    seed = 42

    # regression data
    reg_split = reg_dataset["train"].train_test_split(val_size, seed=seed)
    reg_dataset["train"] = reg_split["train"]
    reg_dataset["val"] = reg_split["test"]

    # initialize regressor model
    reg = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    # reg = BertweetRegressor()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU.")
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
    reg.to(device)

    # train the regressor
    train(reg, train_data=reg_dataset["train"], val_data=reg_dataset["val"])
