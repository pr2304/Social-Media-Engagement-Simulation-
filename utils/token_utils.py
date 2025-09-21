from transformers import DistilBertTokenizer
import pandas as pd

def get_max_token_length(csv_file, column="formatted_text", tokenizer_name="distilbert-base-uncased"):
    tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)
    df = pd.read_csv(csv_file)

    max_length, lengths = 0, []
    for text in df[column].tolist():
        encoding = tokenizer(text, truncation=False, return_tensors='pt')
        length = encoding['input_ids'].shape[1]
        lengths.append(length)
        max_length = max(max_length, length)

    return max_length, lengths
