import torch
import torch.nn as nn
from transformers import DistilBertModel

class TextLikesPredictionModel(nn.Module):
    def __init__(self, pretrained_model_name='distilbert-base-uncased'):
        super(TextLikesPredictionModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained(pretrained_model_name)
        self.regression_head = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 180),
            nn.ReLU(),
            nn.Linear(180, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:, 0, :]
        return self.regression_head(hidden_state)
