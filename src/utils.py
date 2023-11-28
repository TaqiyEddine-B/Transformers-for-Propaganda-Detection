import random
import re

import jsonlines
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm


def preprocess(text):
    """ Preprocess """

    stopwords = []
    extra_stopwords = [
        # 'LINK',
        # 'USER',
        'RT',
        '@']
    stopwords = list(extra_stopwords)

    pattern = r'(?i)\b(?:' + '|'.join(re.escape(word) for word in stopwords) + r')\b'
    text = re.sub(pattern, '', text).strip()

    return text


def read_data_1a(train_file:str,preprocessing:bool):
    """ Read data for the task 1A"""

    train_texts = []
    train_labels = []
    train_types = []
    train_ids = []

    with jsonlines.open(train_file) as train_f:
        for obj in tqdm(train_f, desc="Processing", unit="line"):
            label = 1 if obj["label"]=="true" else 0
            train_labels.append(label)
            train_texts.append(preprocess(obj['text']) if preprocessing else obj['text'] )
            train_ids.append(obj['id'])
            types = 1 if obj["type"]=="tweet" else 0
            train_types.append(types)

    return train_labels,train_texts,train_types,train_ids


def read_data_1b(train_file,preprocessing:bool,unique_labels):
    """ Read data for the task 1B"""
    train_texts = []
    train_labels = []
    train_types = []

    mlb = MultiLabelBinarizer(classes= list(unique_labels))


    train_ids = []

    with jsonlines.open(train_file) as train_f:
        for obj in tqdm(train_f, desc="Processing", unit="line"):
            doc_id = str(obj["id"])
            labels = obj["labels"]
            labels = mlb.fit_transform([set(labels)]).tolist()[0]
            train_labels.append(labels)

            train_texts.append(
                preprocess(obj["text"])
                if preprocessing
                else obj["text"]
            )
            types = 1 if obj["type"] == "tweet" else 0
            train_types.append(types)

            train_ids.append(obj['id'])


    return train_texts,train_labels,train_types,train_ids

def evaluate(pred_labels, gold_labels, subtask, techniques=None):
    """
    Evaluates the predicted classes w.r.t. a gold file.
    Metrics are:  macro_f1 nd micro_f1
    :param pred_labels: a dictionary with predictions,
    :param gold_labels: a dictionary with gold labels.
    """
    pred_values, gold_values = pred_labels, gold_labels

    # We are scoring for subtask 1B
    if subtask == "1B":
        mlb = MultiLabelBinarizer()
        mlb.fit([techniques])
        gold_values = mlb.transform(gold_values)
        # pred_values = mlb.transform(pred_values)

    micro_f1 = f1_score(gold_values, pred_values, average="micro")
    macro_f1 = f1_score(gold_values, pred_values, average="macro")

    return micro_f1, macro_f1

def set_seed(seed):
    """
    Set the seed for random number generation in Python's random, numpy, and torch libraries.
    If a GPU is available, it also sets the seed for random number generation on the GPU.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_output_file(result_path:str,test_ids,pred_labels):
    """ Save output file on disk"""
    with open(result_path, "w") as file:
        file.write("id\tlabel\n")
        for (id,pred) in zip(test_ids,pred_labels):
            label = "true" if pred==1 else "false"
            file.write(str(id)+"\t"+str(label)+"\n")
