from transformers import pipeline
import torch
import pandas as pd

device = 0 if torch.cuda.is_available() else -1  # 0 = GPU, -1 = CPU

text = """Dear Amazon, last week I ordered an Optimus Prime action figure
from your online store in Germany. Unfortunately, when I opened the package,
I discovered to my horror that I had been sent an action figure of Megatron
instead! As a lifelong enemy of the Decepticons, I hope you can understand my
dilemma. To resolve the issue, I demand an exchange of Megatron for the
Optimus Prime figure I ordered. Enclosed are copies of my records concerning
this purchase. I expect to hear from you soon. Sincerely, Bumblebee."""

# classifier = pipeline("text-classification",
#                       device=device)
# outputs = classifier(text)
# df = pd.DataFrame(outputs)
# print(df)

## named entity recognition (NER)
# ner_tagger = pipeline("ner", aggregation_strategy="simple",
#                       device=device)
#
# outputs = ner_tagger(text)
# df = pd.DataFrame(outputs)
# print(df)

# ## question-answering
# reader = pipeline("question-answering",
#                   device=device)
# # question = "What does the customer want?"
# question = "Where is the customer from?"
# outputs = reader(question=question, context=text)
# df = pd.DataFrame([outputs])
# print(df)


## summarization of text
# summarizer = pipeline("summarization",
#                       device=device)
# outputs = summarizer(text, max_length=56, clean_up_tokenization_spaces=True)
# print(outputs[0]['summary_text'])


# ## Translation
# translator = pipeline("translation_en_to_hi",
#                       model="Helsinki-NLP/opus-mt-en-hi",
#                       device=device)
# outputs = translator(text, clean_up_tokenization_spaces=True, min_length=100)
# print(outputs[0]['translation_text'])


## Text Generation
generator = pipeline("text-generation",
                     device=device)
response = "Dear Bumblebee, I am sorry to hear that your order was mixed up."
prompt = text + "\n\nCustomer service response:\n" + response
outputs = generator(prompt, max_length=200)
print(outputs[0]['generated_text'])