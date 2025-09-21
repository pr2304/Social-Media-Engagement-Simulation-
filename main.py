import torch
from torch.utils.data import DataLoader
from models.clip_model import load_clip_model, average_logits
from models.bert_model import TextLikesPredictionModel
from datasets.clip_dataset import CLIPDataset
from datasets.bert_dataset import TextLikesDataset
from utils.train_utils import compute_loss, custom_collate_fn
from transformers import DistilBertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example: Load CLIP
clip_model, processor = load_clip_model(device=device)
clip_dataset = CLIPDataset(csv_file="/kaggle/input/corpus2/pre_proc_train.csv", processor=processor)
clip_loader = DataLoader(clip_dataset, batch_size=16, shuffle=True, 
                         collate_fn=lambda b: custom_collate_fn(b, processor.tokenizer.pad_token_id))

# Example: Load BERT
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_dataset = TextLikesDataset("/kaggle/input/corpus2/pre_proc_train.csv", tokenizer)
bert_loader = DataLoader(bert_dataset, batch_size=16, shuffle=True)

bert_model = TextLikesPredictionModel().to(device)
