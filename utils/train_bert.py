import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from models.bert_model import TextLikesPredictionModel
from datasets.bert_dataset import TextLikesDataset

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer & Dataset
csv_file = "/kaggle/input/corpus2/pre_proc_train.csv"
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_dataset = TextLikesDataset(csv_file=csv_file, tokenizer=tokenizer, max_length=128)
bert_loader = DataLoader(bert_dataset, batch_size=16, shuffle=True)

# Model & Optimizer
bert_model = TextLikesPredictionModel().to(device)
optimizer = torch.optim.Adam(bert_model.parameters(), lr=1e-5)
criterion = nn.MSELoss()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    bert_model.train()
    total_loss = 0.0

    for batch in bert_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['target'].to(device)

        optimizer.zero_grad()
        outputs = bert_model(input_ids, attention_mask)
        loss = criterion(outputs.squeeze(), targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"[BERT] Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(bert_loader):.4f}")
