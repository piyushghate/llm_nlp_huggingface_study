from transformers import AutoTokenizer
# from transformers import DistilBertTokenizer

from datasets import load_dataset

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
# distilbert_tokenizer = DistilBertTokenizer.from_pretrained(model_ckpt)

text = "Tokenizing text is a core task of NLP."

encoded_text = tokenizer(text)
# encoded_text = distilbert_tokenizer(text)
print(encoded_text)

tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
print(tokens)
print(tokenizer.convert_tokens_to_string(tokens))


print("*****************Datasets****************************")

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

emotions = load_dataset("emotion")

print(tokenize(emotions["train"][:2]))

emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

print(emotions_encoded["train"].column_names)
