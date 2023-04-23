import numpy as np
# from sklearn.metrics import r2_score
import torch
import torch.nn as nn
from datasets import load_dataset, Dataset
import transformers
from transformers import AutoTokenizer, AutoModel, AdamW, AutoModelForSequenceClassification, \
    get_linear_schedule_with_warmup, BertTokenizer, BertModel
from tqdm import tqdm
# Bertweet regressor
class BertweetRegressor(nn.Module):

    def __init__(self, drop_rate=0.2, freeze_bertweet=False):
        super(BertweetRegressor, self).__init__()
        D_in, D_out = 768, 3

        self.bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
        self.regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(D_in, D_out))
#         self.double()

    def forward(self, input_ids, attention_masks):
        outputs = self.bertweet(input_ids, attention_masks)
        class_label_output = outputs[1]
        outputs = self.regressor(class_label_output)
        return outputs

    
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
def evaluate(model, test_data: Dataset, batch_size: int = 32):
    model.eval()
    losses = []
    with torch.no_grad():
        for i in tqdm(range(0, len(test_data), batch_size)):
            batch = test_data[i:i + batch_size]
            input_ids, attention_mask = torch.tensor(batch["input_ids"]).to(device), torch.tensor(batch["attention_mask"]).to(device)
            outputs = model(input_ids, attention_mask)
            batch_labels = torch.tensor(np.array([batch["V"], batch["A"], batch["D"]]).T).float().to(device)
            loss_function = nn.MSELoss(reduction="sum")
            loss = loss_function(outputs, batch_labels)
            losses.append(loss)
        return sum(losses), cal_r2_score(outputs, batch_labels)

# trainer
def train(BertweetRegressor, train_data: Dataset, val_data: Dataset,
          batch_size: int = 32, max_epochs: int = 5,
          file_path: str = "checkpoints"):
    adam = AdamW(BertweetRegressor.parameters(), lr=5e-5, eps=1e-8)
    loss_function = nn.MSELoss(reduction="sum")
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
            input_ids, attention_mask = torch.tensor(batch["input_ids"]).to(device), torch.tensor(batch["attention_mask"]).to(device)
            logits = BertweetRegressor(input_ids, attention_mask)
            batch_labels = torch.tensor(np.array([batch["V"], batch["A"], batch["D"]]).T).float().to(device)
            loss = loss_function(logits, batch_labels)
            adam.zero_grad()
            loss.backward()
#             print(loss)
            adam.step()
        # Test on validation data
        print("Evaluating on validation data...")
        val_loss, r2 = evaluate(BertweetRegressor, val_data, batch_size=batch_size)
        print("Validation loss: {:.3f}, r2 score: {}".format(val_loss, r2))
        r_scores.append(r2)
        torch.save(BertweetRegressor.state_dict(), "{}/epoch{}.pt".format(file_path, epoch))
#     print(r_scores)
    r_scores = torch.tensor(r_scores)
    print("Best val achieved at epoch {}, with r2 score{}".format(torch.argmax(r_scores), torch.max(r_scores)))


# def init_trainer(model_name, train_data, val_data):
#
#     training_args = TrainingArguments(output_dir="checkpoints",
#                                       evaluation_strategy="epoch",
#                                       num_train_epochs=5)
#
#     # define loss function
#     def compute_metrics(eval_pred):
#         logits, labels = eval_pred
#         print(labels)
#         references = np.array([[tup[0], tup[1], tup[2]] for tup in labels])
#         assert logits.shape == references.shape
#         metric = evaluate.load("mse")
#         return metric.compute(predictions=logits, references=references)
#
#     model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3, problem_type='regression')
#     return Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_data,
#         eval_dataset=val_data,
#         compute_metrics=compute_metrics,
#     )

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
    clf_dataset = load_dataset("csv", data_files={"train": "sar_and_meta_train.csv", "test": "sar_and_meta_test.csv"})
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    reg_dataset["train"] = preprocess_data(reg_dataset["train"], tokenizer)
#     reg_dataset["train"]["input_ids"] = torch.tensor(reg_dataset["train"]["input_ids"])
    clf_dataset["train"] = preprocess_data(clf_dataset["train"], tokenizer)
#     clf_dataset["train"]["input_ids"] = torch.tensor(clf_dataset["train"]["input_ids"])
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
    
#     print(reg_dataset["train"][:2])
    # initialize regressor model
    reg = BertweetRegressor()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU.")
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
    reg.to(device)

    # train the regressor
    train(reg, train_data=reg_dataset["train"], val_data=reg_dataset["val"])

    # trainer = init_trainer(model_name, emo_dataset["train"], emo_dataset["val"])
    # trainer.train()
