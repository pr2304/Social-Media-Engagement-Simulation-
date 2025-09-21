import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from utils.image_utils import load_image, create_placeholder_image

class CLIPDataset(Dataset):
    def __init__(self, csv_file, processor, ref_img=None):
        self.data = pd.read_csv(csv_file)
        self.processor = processor
        self.ref_img = ref_img or create_placeholder_image()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_url = self.data.iloc[idx]['full_image_url']
        text = self.data.iloc[idx]['formatted_text']
        target = self.data.iloc[idx]['likes']

        image = load_image(img_url)
        if list(image.getdata()) == list(self.ref_img.getdata()):
            return None

        encoding = self.processor(text=[text], images=image, return_tensors="pt", padding=True, truncation=True)
        return encoding['input_ids'].squeeze(0), encoding['pixel_values'].squeeze(0), target
