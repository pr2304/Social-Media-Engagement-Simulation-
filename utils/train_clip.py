import torch
from torch.utils.data import DataLoader
from models.clip_model import load_clip_model, average_logits
from datasets.clip_dataset import CLIPDataset
from utils.train_utils import compute_loss, custom_collate_fn

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CLIP model and processor
clip_model, processor = load_clip_model(device=device)
clip_model.train()

# Dataset & DataLoader
csv_file = "/kaggle/input/corpus2/pre_proc_train.csv"
clip_dataset = CLIPDataset(csv_file=csv_file, processor=processor)
clip_loader = DataLoader(
    clip_dataset, 
    batch_size=16, 
    shuffle=True, 
    collate_fn=lambda b: custom_collate_fn(b, processor.tokenizer.pad_token_id)
)

# Optimizer
optimizer = torch.optim.Adam(clip_model.parameters(), lr=1e-5)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0.0
    for batch in clip_loader:
        if batch is None:
            continue

        input_ids, pixel_values, targets = batch
        input_ids, pixel_values, targets = input_ids.to(device), pixel_values.to(device), targets.to(device).float()

        optimizer.zero_grad()
        outputs = clip_model(input_ids=input_ids, pixel_values=pixel_values)
        logits_per_image = outputs.logits_per_image

        avg_logits = average_logits(logits_per_image)   # [batch_size]
        loss = compute_loss(avg_logits, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"[CLIP] Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(clip_loader):.4f}")
