import numpy as np
import evaluate
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoModel, AdamW, AutoModelForSequenceClassification, \
    get_linear_schedule_with_warmup, BertTokenizer, BertModel
from transformers import Trainer, TrainingArguments


def init_trainer(model_name, train_data, val_data):

    training_args = TrainingArguments(output_dir="checkpoints",
                                      evaluation_strategy="epoch",
                                      num_train_epochs=5)

    # define loss function
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        references = np.array([[tup[0], tup[1], tup[2]] for tup in labels])
        assert logits.shape == references.shape
        metric = evaluate.load("mse")
        return metric.compute(predictions=logits, references=references)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,
    )

def preprocess_data(dataset, tokenizer):
    dataset.map(lambda x: tokenizer(x['text'],
                                    add_special_tokens=True,
                                    padding="max_length",
                                    max_length=128,
                                    truncation="longest_first",
                                    return_attentiton_mask=True,
                                    ))
    return dataset


if __name__ == '__main__':
    model_name = "vinai/bertweet-base"

    # load dataset
    emo_dataset = load_dataset("csv", data_files={"train": "norm_emobank_train.csv", "test": "norm_emobank_test.csv"})
    sar_dataset = load_dataset("csv", data_files={"train": "sarcasm_train.csv", "test": "sarcasm_test.csv"})
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    emo_dataset["train"] = preprocess_data(emo_dataset["train"], tokenizer)
    sar_dataset["train"] = preprocess_data(sar_dataset["train"], tokenizer)

    # split training set into traindev
    val_size = 0.1
    seed = 42
    emo_split = emo_dataset["train"].train_test_split(val_size, seed=seed)
    emo_dataset["train"] = emo_split["train"]
    emo_dataset["val"] = emo_split["test"]

    sar_split = sar_dataset["train"].train_test_split(val_size, seed=seed)
    sar_dataset["train"] = sar_split["train"]
    sar_dataset["val"] = sar_split["test"]

    trainer = init_trainer(model_name, emo_dataset["train"], emo_dataset["val"])
    trainer.train()


