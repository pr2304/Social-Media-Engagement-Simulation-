from PIL import Image
import requests
from io import BytesIO

def create_placeholder_image(size=(224, 224)):
    return Image.new('RGB', size, color='black')

def load_image(image_url, size=(224, 224)):
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content)).convert("RGB")
            return image.resize(size)
        else:
            return create_placeholder_image(size)
    except Exception:
        return create_placeholder_image(size)
