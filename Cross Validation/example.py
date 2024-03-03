from model import TransformerEncoder, WordPiece
from cross_validation import cross_validation, TorchLoader

import torch
from torch import optim
from torch.utils.data import Dataset

import pandas as pd


# Coverting the DataFrame into a Pytorch Dataset
class FinancialNewsDataset(Dataset):
    def __init__(self, dataframe, classes):
        super().__init__()

        self.samples = [(dataframe["En_Text"][i], dataframe["Label"][i]) for i in range(len(dataframe))]
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_classes = {i: c for i, c in enumerate(self.classes)}
    
    def __getitem__(self, index):
        if isinstance(index, slice):
            return [(torch.tensor(sample[0], dtype=torch.long), sample[1]) for sample in self.samples[index]] # List (Tuple (Tensor, Int) )
        return (torch.tensor(self.samples[index][0], dtype=torch.long), self.samples[index][1])               # Tuple (Tensor, Int)

    def __len__(self):
        return len(self.samples)


# Setting default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading/Processing the Dataset
df = pd.read_csv("sms_spam.csv", encoding="latin-1")[["v1", "v2"]]
df.rename(columns={"v1": "Label", "v2": "Text"}, inplace=True)
df["Label"] = df["Label"].map({
    "ham": 0,
    "spam": 1
})
df["Cl_Text"] = df["Text"].apply(lambda x: x.lower())

# For the tokenization we are going to use the WordPiece class I've created
corpus = list(df.Cl_Text)

w = WordPiece(corpus=corpus, ntokens=1_000, cleaning=lambda text: text.lower())
w.fit()
vocab_size = len(w.vocab_l)

# Getting max text length for padding
max_len = df["Cl_Text"].str.len().max() + 1 # due to [SEP] at the end of each sentence

df["En_Text"] = df["Cl_Text"].apply(lambda x: [w.vocab_d["[CLS]"]] + w.encode(text=x, npad=max_len))

max_len += 1 # due to the [CLS] token


ds = FinancialNewsDataset(df, ["ham", "spam"])
classes = ds.classes
class_to_idx = ds.class_to_idx
idx_to_classes = ds.idx_to_classes

train_size = int(len(ds) * 0.95)
train_ds = ds[:train_size]
test_ds = ds[train_size:]


EMBED_SIZE = 100
NUM_LAYERS = 2
SCALE_EMBEDS = 1
NUM_HEADS = 2
DROPOUT = 0.1

model = TransformerEncoder(
    vocab_size = vocab_size,
    max_length = max_len,
    embed_size = EMBED_SIZE,
    num_layers = NUM_LAYERS,
    num_heads = NUM_HEADS,
    scale_embeds = SCALE_EMBEDS,
    dropout = DROPOUT,
    output_size = 2,
    device = device
).to(device)


opt = optim.AdamW(model.parameters(), lr=1e-3)

EPOCHS = 2

res = model.fit_cv(EPOCHS, train_ds, opt, cross_validation, valid_prop=0.05, batch_size=1)

print(model.evaluate(TorchLoader(test_ds, batch_size=32, shuffle=False)))
