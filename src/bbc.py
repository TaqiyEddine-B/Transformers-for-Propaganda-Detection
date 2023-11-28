
import argparse
import logging
import logging.handlers
import warnings
from typing import Optional, Tuple, Union

import jsonlines
import torch
# import wandb
from balanced_loss import Loss
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import (
    BertForSequenceClassification,
    BertModel,
    BertPreTrainedModel,
    BertTokenizer,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from src.loss import FocalLoss


class BertBinaryClassifier:
    def __init__(self, model_path, device="cuda" if torch.cuda.is_available() else "cpu",archi="monotask"):
        if archi=="monotask":
          self.model = BertForSequenceClassification.from_pretrained(model_path)
        elif archi=="multitask":
          self.model = BertForMultitaskClassification.from_pretrained(model_path)
        elif archi=="feature":
          self.model = BertForClassification_feature.from_pretrained(model_path)
        self.archi = archi
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.device = torch.device(device)
        self.model.to(self.device)

    def tokenize_texts(self, texts,labels2=None):
        encoded_texts = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        input_ids = encoded_texts["input_ids"].to(self.device)
        attention_mask = encoded_texts["attention_mask"].to(self.device)
        if labels2 is not None and self.archi=="feature":
            labels2 = torch.tensor(labels2).to(self.device)

        return input_ids, attention_mask, labels2 if not None else None

    def train(self, texts, labels, labels2=None, batch_size=8, epochs=4, learning_rate=2e-5,loss="cross-entropy"):
        input_ids, attention_mask, _ = self.tokenize_texts(texts)
        labels = torch.tensor(labels).to(self.device)
        if self.archi!="monotask":
          labels2 = torch.tensor(labels2).to(self.device)
          dataset = TensorDataset(input_ids, attention_mask, labels,labels2)
        else:
          dataset = TensorDataset(input_ids, attention_mask, labels)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        print(f" learning_rate ->  {learning_rate}")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        if loss=="cross-entropy":
            loss_function = torch.nn.CrossEntropyLoss()
        elif loss=="focal":
            loss_function = FocalLoss()
        elif loss == "focal_balanced_loss":
            loss_function = Loss(
                loss_type="focal_loss")

        for epoch in tqdm(range(epochs), desc="Running epoch "):
            self.model.train()
            total_loss = 0.0
            for batch in dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                if self.archi!='monotask':
                  input_ids, attention_mask, labels , labels2 = batch
                else:
                  input_ids, attention_mask, labels = batch


                optimizer.zero_grad()

                if self.archi=="monotask":
                  outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                  logits = outputs.logits
                  loss = loss_function(logits.view(-1, 2), labels.view(-1))

                elif self.archi=="multitask":
                  outputs, logits2 = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                  logits = outputs.logits
                  loss1 = loss_function(logits.view(-1, 2), labels.view(-1))
                  loss2 = loss_function(logits2.view(-1, 2), labels2.view(-1))
                  loss=loss1+loss2

                elif self.archi=="feature":
                  outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, feature_type=labels2)
                  logits = outputs.logits
                  loss = loss_function(logits.view(-1, 2), labels.view(-1))

                # wandb.log({"loss": loss})

                total_loss += loss.item()

                loss.backward()
                optimizer.step()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{epochs} - Average Loss: {avg_loss}")

    def predict(self, texts,feature_type):
        input_ids, attention_mask, feature_type_tensor = self.tokenize_texts(texts,feature_type)
        with torch.no_grad():
          if(self.archi=='feature' and feature_type_tensor is not None):
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask,feature_type=feature_type_tensor)
          elif(self.archi=="multitask"):
            logits,_ = self.model(input_ids=input_ids, attention_mask=attention_mask)
          elif(self.archi=='monotask'):
            logits= self.model(input_ids=input_ids, attention_mask=attention_mask)
          probs = torch.softmax(logits.logits, dim=-1)
          predicted_labels = torch.argmax(probs, dim=-1)
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
        self.classifier2 = nn.Linear(config.hidden_size, config.num_labels)

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



class BertForClassification_feature(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size+1, config.num_labels)

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
        feature_type: Optional[torch.Tensor] = None,
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
        logits = self.classifier(torch.cat([pooled_output, feature_type.unsqueeze(dim=1)], dim=1))

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
        )