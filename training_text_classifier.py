# from transformers import AutoModel
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
# from transformers import AutoTokenizer
import torch
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


model_ckpt = "distilbert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
logging_steps = len(emotions_encoded["train"]) // batch_size
model_name = f"{model_ckpt}-finetuned-emotion"

emotions = load_dataset("emotion")

emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

# tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
# model = AutoModel.from_pretrained(model_ckpt).to(device)
#
#
# text = "this is a test"
# inputs = tokenizer(text, return_tensors="pt")
# print(f"Input tensor shape: {inputs['input_ids'].size()}")


#########################################
### Fine-Tuning Transformers #############
######################################

num_labels = 6
model_ckpt = "distilbert-base-uncased"

model = (AutoModelForSequenceClassification
.from_pretrained(model_ckpt, num_labels=num_labels)
.to(device))

batch_size = 64
logging_steps = len(emotions_encoded["train"]) // batch_size
model_name = f"{model_ckpt}-finetuned-emotion"
training_args = TrainingArguments(output_dir=model_name,
    num_train_epochs=2,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    disable_tqdm=False,
    logging_steps=logging_steps,
    push_to_hub=True,
    log_level="error")