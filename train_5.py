import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import DistilBertTokenizer, DistilBertModel
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from torch.amp import autocast, GradScaler  


df = pd.read_csv('XSS_dataset.csv')


label_counts = df['Label'].value_counts()
print(label_counts)


class XSSDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        sentence = str(self.sentences[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'sentence_text': sentence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class XSSDetector(nn.Module):
    def __init__(self, n_classes):
        super(XSSDetector, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 128)
        self.relu = nn.ReLU()
        self.out = nn.Linear(128, n_classes)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = bert_output.last_hidden_state[:, 0, :]
        output = self.drop(output)
        output = self.fc1(output)
        output = self.relu(output)
        return self.out(output)


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
MAX_LEN = 256
BATCH_SIZE = 64
RANDOM_SEED = 42


sentences = df['Sentence'].values
labels = df['Label'].values
train_sentences, val_sentences, train_labels, val_labels = train_test_split(
    sentences, labels, test_size=0.1, random_state=RANDOM_SEED, stratify=labels
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = torch.tensor([label_counts[0] / len(df), label_counts[1] / len(df)], dtype=torch.float32, device=device)


train_dataset = XSSDataset(train_sentences, train_labels, tokenizer, MAX_LEN)
val_dataset = XSSDataset(val_sentences, val_labels, tokenizer, MAX_LEN)
train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)


model = XSSDetector(n_classes=2).to(device)
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)  
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


scaler = GradScaler()


def train_epoch(model, data_loader, loss_fn, optimizer, device, n_examples, scaler):
    model.train()
    losses = []
    correct_predictions = 0

    for d in tqdm(data_loader, desc="Training Batches"):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        optimizer.zero_grad()

        
        with autocast(device_type='cuda'):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model.eval()
    losses = []
    correct_predictions = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for d in tqdm(data_loader, desc="Validation Batches"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            
            with autocast(device_type='cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)
                loss = loss_fn(outputs, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    return correct_predictions.double() / n_examples, np.mean(losses)


EPOCHS = 10
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer, device, len(train_dataset), scaler)
    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(model, val_data_loader, loss_fn, device, len(val_dataset))
    print(f'Val loss {val_loss} accuracy {val_acc}')

    scheduler.step()


torch.save(model.state_dict(), 'xss_detection_model.pth')

