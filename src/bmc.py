import logging
import logging.handlers

import torch


import argparse
import logging
import logging.handlers
import warnings
from typing import Optional, Tuple, Union

import jsonlines
import torch
from balanced_loss import Loss
from loss import FocalLoss
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import (
    BertForSequenceClassification,
    BertModel,
    BertPreTrainedModel,
    BertTokenizer,
)
from transformers.modeling_outputs import SequenceClassifierOutput

import wandb


class BertMultilabelClassifier:
    def __init__(self, model_path, num_classes,device="cuda" if torch.cuda.is_available() else "cpu"):

        self.model = BertForMultitaskClassification.from_pretrained(model_path,num_labels=num_classes, problem_type="multi_label_classification")
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.num_classes = num_classes

    def tokenize_texts(self, texts):
        encoded_texts = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        input_ids = encoded_texts["input_ids"].to(self.device)
        attention_mask = encoded_texts["attention_mask"].to(self.device)

        return input_ids, attention_mask

    def train(self, texts, labels,labels_types, batch_size=8, epochs=4, learning_rate=2e-5):
        input_ids, attention_mask = self.tokenize_texts(texts)

        labels = torch.tensor(labels).to(torch.float).to(self.device)
        labels_types = torch.tensor(labels_types).to(torch.long).to(self.device)

        dataset = TensorDataset(input_ids, attention_mask, labels,labels_types)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        loss_function = FocalLoss()
        # loss_function = BCEWithLogitsLoss()
        for epoch in tqdm(range(epochs), desc="Running epoch "):
            self.model.train()
            total_loss = 0.0
            for batch in dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, labels,labels_types = batch

                optimizer.zero_grad()

                outputs,logits2 = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels.float(),labels_types=labels_types.float())
                logits = outputs.logits
                # logits = torch.tensor(logits).to(torch.long).to(self.device)

                loss1 = outputs.loss #loss_function(logits.view(-1, self.num_classes), labels)

                loss_fct = CrossEntropyLoss()
                loss2 = loss_fct(logits2.view(-1, 2), labels_types.view(-1))
                loss = loss1 + loss2
                # loss = loss_function(logits, labels)


                wandb.log({"loss": loss})

                total_loss += loss.item()

                loss.backward()
                optimizer.step()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{epochs} - Average Loss: {avg_loss}")

    def predict(self, texts, threshold=0.5):
        input_ids, attention_mask = self.tokenize_texts(texts)
        with torch.no_grad():
          logits,_= self.model(input_ids=input_ids, attention_mask=attention_mask)
          probs = torch.sigmoid(logits.logits)

          predicted_labels = (probs > threshold).int()

          zero_rows = (predicted_labels.sum(dim=1) == 0)
          predicted_labels[zero_rows, -3] = 1

        return predicted_labels, probs


class BertForMultitaskClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier2 = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        labels_types: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) :
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits2 = self.classifier2(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ), logits2

