import os
from torch.nn.utils.clip_grad import clip_grad_norm
import torch
import torch.nn as nn
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModel, AdamW, AutoModelForSequenceClassification, \
    get_linear_schedule_with_warmup, BertTokenizer, BertModel
from tqdm import tqdm
from transformers.adapters import RobertaAdapterModel
from sklearn.metrics import f1_score

# implementation of FL_pooler class
class FL_pooler(nn.Module):
    def __init__(self, drop_rate=0.2):
        super(FL_pooler, self).__init__()
        D_in, D_out = 3, 1
        D_pooler_out = 768
        # get file path of fine-tuned regressor
        p_V_path, p_A_path, p_D_path = "checkpoints/best_res/epoch8@sid129682.pt", "checkpoints/best_res/epoch9@sid129776.pt", "checkpoints/best_res/epoch8@sid129709.pt"
        self.v_bert = RobertaAdapterModel.from_pretrained("vinai/bertweet-base")
        self.a_bert = RobertaAdapterModel.from_pretrained("vinai/bertweet-base")
        self.d_bert = RobertaAdapterModel.from_pretrained("vinai/bertweet-base")
        self.v_bert.load_state_dict(torch.load(p_V_path), strict=False)
        self.a_bert.load_state_dict(torch.load(p_A_path), strict=False)
        self.d_bert.load_state_dict(torch.load(p_D_path), strict=False)
        # freeze the parameters
        for param in self.v_bert.parameters():
           param.requires_grad = False
        for param in self.a_bert.parameters():
           param.requires_grad = False
        for param in self.d_bert.parameters():
           param.requires_grad = False
        
#         # add adapters and activate them
#         self.v_bert.add_adapter("v_adapter")
#         self.a_bert.add_adapter("a_adapter")
#         self.d_bert.add_adapter("d_adapter")
#         # Activate the adapter
#         self.v_bert.train_adapter("v_adapter")
#         self.a_bert.train_adapter("a_adapter")
#         self.d_bert.train_adapter("d_adapter")
#         print("adapters activated!")
        self.pooler = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(D_in, D_out))
        self.clf = nn.Linear(D_pooler_out, 3)
        print("initialization completed!")

    def forward(self, input_ids, attention_masks):
        # get outputs for three models
        v_outputs = self.v_bert(input_ids, attention_masks)
        a_outputs = self.a_bert(input_ids, attention_masks)
        d_outputs = self.d_bert(input_ids, attention_masks)
        # concat the output and align the dimensions
        concated = torch.cat((v_outputs[1], a_outputs[1], d_outputs[1]), dim=1)
        concated = torch.reshape(concated, (concated.size()[0], 3, concated.size()[1] // 3))
        concated = concated.transpose(-2, -1)
        # for debugging
#         print(concated.size())
        output = self.pooler(concated)
        # get logits
        logits = self.clf(output.squeeze())
        return logits


# evaluate model performance (acc)
def evaluate(model, test_data: Dataset, batch_size: int = 32):
    model.eval()
    with torch.no_grad():
        total_correct = 0
        losses = []
        scores = []
        loss_function = nn.CrossEntropyLoss()
        for i in tqdm(range(0, len(test_data), batch_size)):
            batch = test_data[i:i + batch_size]
            input_ids, attention_mask = torch.tensor(batch["input_ids"]).to(device), \
                                        torch.tensor(batch["attention_mask"]).to(device)
            # get classifier logit scores
            outputs = model(input_ids, attention_masks=attention_mask)
            # ground truth labels
            test_labels = torch.tensor(batch["label"]).to(device)
            probs = nn.functional.softmax(outputs)
            # get model output labels
            labels = torch.argmax(probs, dim=1)
            # calculate cross entropy loss
            loss = loss_function(probs, test_labels)
            losses.append(loss)
            # calculate accuracy
            correct = torch.tensor(labels == test_labels)
            total_correct += sum(correct)
            # compute f1 score
            f1 = f1_score(test_labels, labels, 'weighted')
            scores.append(f1)
        acc = total_correct / len(test_data)
        return acc, torch.mean(torch.tensor(losses)), torch.mean(torch.tensor(scores))


# trainer
def train(model, train_data: Dataset, val_data: Dataset,
          batch_size: int=32, max_epochs: int=10,
          file_path: str="checkpoints/pooler_clf", checkpoint_path: str="checkpoints/pooler_checkpts", clip_value: int=2):
    # switch to training mode
    model.train()
    # check trainable params
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params: {}".format(num_trainable_params))
    # initialize optimizer
    lr, lr_mul = 8e-5, 1
    # weight_decay = 1e-5
    eps = 1e-8
    adam = AdamW(model.parameters(),
                 # [{'params': bert_param},
                 # {'params': reg_param, 'lr': lr * lr_mul, 'weight_decay': weight_decay}],
                 lr=lr,
                 eps=eps,
                 # weight_decay=weight_decay,
                 )
    # initialize scheduler
#     total_steps = len(train_data) * max_epochs
#     scheduler = get_linear_schedule_with_warmup(adam,
#                                                 num_warmup_steps=0,
#                                                 num_training_steps=total_steps)
    # assign loss function
    loss_function = nn.CrossEntropyLoss()
    # load checkpoints
    file = [f for f in os.listdir(checkpoint_path)]
    EPOCH = 0
    if "clf_checkpt.pt" in file:
#         checkpoint = torch.load(os.path.join(checkpoint_path, "clf_checkpt_vua.pt"))
#         model.load_state_dict(checkpoint['model_state_dict'])
#         adam.load_state_dict(checkpoint['optimizer_state_dict'])
#         EPOCH = checkpoint['epoch']
#         loss = checkpoint['loss']
#         print("checkpoint found at epoch:{}".format(EPOCH+1))
        EPOCH += 1
        
    
    # store historical accs
    val_accs = []
    avg_f1s = []
    best_acc = 0
    for epoch in range(max_epochs):
        print("Epoch {} of {}".format(epoch+1+EPOCH, max_epochs+EPOCH))
        # Training code
        print("Training...")
        for i in tqdm(range(0, len(train_data), batch_size)):
            batch = train_data[i:i + batch_size]
            # calculate loss and do SGD
            input_ids, attention_mask = torch.tensor(batch["input_ids"]).to(device), torch.tensor(batch["attention_mask"]).to(device)
            batch_labels = torch.tensor(batch["label"]).to(device)
            logits = model(input_ids, attention_mask)
            loss = loss_function(logits, batch_labels)
            adam.zero_grad()
            loss.backward()
            # prevent gradient vanishing
#             clip_grad_norm(model.parameters(), clip_value)
            adam.step()
#             scheduler.step()
        # Test on validation data
        print("Evaluating on validation data...")
        val_acc, loss, avg_f1 = evaluate(model, val_data)
        print("Validation acc: {:.3f}, cross entropy loss: {:.3f}, F1 score: {:.3f}".format(val_acc, loss, avg_f1))
        val_accs.append(val_acc)
        avg_f1s.append(avg_f1)
        # save the checkpoint of the current epoch
#         torch.save({
#             'epoch': epoch+1+EPOCH,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': adam.state_dict(),
#             'loss': loss,
#             }, os.path.join(checkpoint_path, "clf_checkpt_vua.pt"))
        
        if val_acc.item() > best_acc:
            best_acc = val_acc.item()
            torch.save(model.state_dict(),
                       "{}/best_acc_sid{}.pt".format(file_path, os.environ['SLURM_JOB_ID']))
    val_accs = torch.tensor(val_accs)
    print("Best val acc achieved at epoch {}, with acc {}, f1 score {}, slurm_job_id: {}".format(torch.argmax(val_accs)+1+EPOCH,
                                                                                     torch.max(val_accs), torch.max(avg_f1s), 
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
    clf_dataset = load_dataset("csv", data_files={"train": "train_with_vua.csv", "test": "test_with_vua.csv"})
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
    model.load_state_dict(torch.load("checkpoints/pooler_clf/best_acc_sid133500.pt"))
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
