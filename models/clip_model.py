import torch
from transformers import CLIPProcessor, CLIPModel

def load_clip_model(model_name="openai/clip-vit-base-patch16", device="cuda"):
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor

def average_logits(logits):
    if logits.dim() != 2:
        raise ValueError("Logits must be 2D")
    return torch.mean(logits, dim=1)