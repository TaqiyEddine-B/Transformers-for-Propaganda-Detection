import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel

class MultilabelBERTClassifier(nn.Module):
    def __init__(self, num_labels, model_name='bert-base-uncased'):
        super(MultilabelBERTClassifier, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])
        return self.sigmoid(logits)

    def tokenize(self, texts, max_length=128):
        return self.tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
    def predict(self, texts, threshold=0.5, max_length=128):
        """
        Predict the labels for a list of texts
        Args:
        - texts (list of str): List of input strings
        - threshold (float): Threshold for classifying a label as positive
        - max_length (int): Max sequence length for tokenization

        Returns:
        - predictions (list of lists of int): Binary predictions for each label
        """
        self.eval()
        with torch.no_grad():
            encodings = self.tokenize(texts, max_length=max_length)
            input_ids, attention_masks = encodings["input_ids"], encodings["attention_mask"]
            logits = self.forward(input_ids, attention_mask=attention_masks)
            predictions = (logits > threshold).int().tolist()
        return predictions
class BERTTrainer:
    def __init__(self, model, texts, labels, learning_rate=2e-5, max_length=128, batch_size=8):
        self.model = model
        self.texts = texts
        self.labels = labels
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.batch_size = batch_size
        self.loss_fn = nn.BCELoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

    def prepare_dataloader(self):
        encodings = self.model.tokenize(self.texts, max_length=self.max_length)
        input_ids, attention_masks = encodings["input_ids"], encodings["attention_mask"]
        labels_tensor = torch.Tensor(self.labels)
        dataset = TensorDataset(input_ids, attention_masks, labels_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def train(self, num_epochs):
        dataloader = self.prepare_dataloader()
        self.model.train()
        for epoch in range(num_epochs):
            for batch in dataloader:
                self.optimizer.zero_grad()
                input_ids, attention_masks, batch_labels = batch
                outputs = self.model(input_ids, attention_mask=attention_masks)
                loss = self.loss_fn(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {epoch+1} loss: {loss.item()}")


# Usage:
# Assuming you have your dataset loaded as lists: texts, labels
texts = ["sample tweet", "sample news article"]
labels = [[1, 0, 0], [0, 1, 1]]  # Assuming 3 possible propaganda techniques

model = MultilabelBERTClassifier(len(labels[0]))
trainer = BERTTrainer(model, texts, labels)
trainer.train(num_epochs=3)



texts = ["sample tweet", "sample news article"]
predictions = model.predict(texts)
print(predictions)  # Example output: [[1, 0, 0], [0, 1, 1]]
