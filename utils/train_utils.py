import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

def compute_loss(logits, targets):
    return F.mse_loss(logits, targets)

def custom_collate_fn(batch, pad_token_id):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    input_ids = [item[0] for item in batch]
    pixel_values = [item[1] for item in batch]
    targets = torch.as_tensor([item[2] for item in batch])

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    pixel_values = torch.stack(pixel_values)

    return input_ids, pixel_values, targets
