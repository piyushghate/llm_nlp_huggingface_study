# from datasets import list_datasets
from datasets import load_dataset
import matplotlib.pyplot as plt
# from huggingface_hub import list_datasets

# all_datasets = list(list_datasets())
# print(f"There are {len(all_datasets)} datasets currently available on the Hub")
# print(f"The first 10 are: {all_datasets[:10]}")

def label_int2str(row):
    return emotions["train"].features["label"].int2str(row)

emotions = load_dataset("emotion")
train_ds = emotions["train"]
print(len(train_ds))
print(train_ds[10017])
print(train_ds.features)
print(train_ds[:5])
print(train_ds["text"][:5])

emotions.set_format(type="pandas")
df = emotions["train"][:]
print(df.head())

df["label_name"] = df["label"].apply(label_int2str)
print(df.head())

df["label_name"].value_counts(ascending=True).plot.barh()
plt.title("Frequency of Classes")
# plt.show()
plt.savefig("plot/Frequency_of_Classes.png")
print("Plot saved as plot.png")

df["Words Per Tweet"] = df["text"].str.split().apply(len)
df.boxplot("Words Per Tweet", by="label_name", grid=False,
showfliers=False, color="black")
plt.suptitle("")
plt.xlabel("")
# plt.show()
plt.savefig("plot/Words_Per_Tweet.png")
print("Plot saved as plot.png")

emotions.reset_format()