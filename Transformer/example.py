from Encoder import TransformerEncoder

import pandas as pd

### Hyperparameters
BATCH_SIZE = 32
EMBED_SIZE = 300
EPOCHS = 500
LR = 3e-4
EVAL_INTERVAL = 50
EVAL_ITERS = 5
NUM_HEADS = 6
NUM_LAYERS = 6
DROPOUT = 0.2
SCALE_EMBEDS = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Loading the Dataset into DataFrame Format
df = pd.read_csv("sms_spam.csv", encoding="latin-1")[["v1", "v2"]] # By default there are another 2 columns that doe't have data
df.rename(columns={"v1": "num_label", "v2": "raw_text"}, inplace=True)

### Creating the Vocabulary (we denote [CLS] as '{' and [PADD] as '}')
vocab = sorted(list(set(c for s in df["raw_text"] for c in s)) + ['{'] + ['}'])
vocab_size = len(vocab)

### Creating Encoder and Decoder Functions
char_to_int = {ch: i for i, ch in enumerate(vocab)}
int_to_char = {i: ch for i, ch in enumerate(vocab)}
encoder = lambda s: [char_to_int[c] for c in s]
decoder = lambda l: "".join([int_to_char[i] for i in l])

### Getting the maximum length of the raw text messages
max_sentence_length = df["raw_text"].str.len().max()

### Preprocessing the Dataset:
# 1. Convert the labels from "ham", "spam" into [1, 0] and [0, 1] respectively
# 2. Add the [CLC] and [PADD] tokens
# 3. Tokenize the text
df["label"] = df["num_label"].map({"ham": [1, 0], "spam": [0, 1]})
df["proc_text"] = df["raw_text"].apply(lambda l: "{" + l + "".join(["}" for _ in range(max_sentence_length - len(l))]))
df["tokenized_text"] = df["proc_text"].apply(encoder)

max_sentence_length += 1 # Due to the [CLC] token

### Splitting the Dataset into Training and Validation Sets
train_size = int(len(df) * 0.9)
train_ds, valid_ds = df.iloc[:train_size], df.iloc[train_size:].reset_index(drop=True) # don't add the previous indexes as a new column

### Creating the Function that Splits the Dataset into Batches
def get_batch(split=None):
    data = train_ds if split == "train" else valid_ds

    idx_xb = torch.randint(len(data), (BATCH_SIZE,)) # Random indexes from the Dataset

    x = torch.stack([torch.tensor(data["tokenized_text"][i.item()], dtype=torch.long) for i in idx_xb])
    y = torch.tensor([data["label"][i.item()] for i in idx_xb], dtype=torch.long)    

    return x.to(device=device, non_blocking=True), y.to(device=device, non_blocking=True)

### Creating a Loss Calculation Function
@torch.inference_mode()
def estimate_loss(model):
    out = {}

    model.eval()
    for split in ["train", "valid"]:
        losses = torch.zeros(EVAL_ITERS)

        # Iterate `EVAL_ITERS` number of times to reduce the noice that each batch contains
        for i in range(EVAL_ITERS):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[i] = loss.item()
        
        out[split] = losses.mean()
    
    model.train()
    return out

### Initializing the Model
model = TransformerEncoder(
    vocab_size = vocab_size,
    max_length = max_sentence_length,
    embed_size = EMBED_SIZE,
    num_layers = NUM_LAYERS,
    num_heads = NUM_HEADS,
    scale_embeds = SCALE_EMBEDS,
    dropout = DROPOUT,
    output_size = 2,
    device = device
).to(device)

### Setting Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

### Training the Model
print("\nStarting Training:")
for epoch in tqdm(range(1, EPOCHS+1)):
    # Evaluating Loss every EVAL_INTERVAL
    if epoch % EVAL_INTERVAL == 0:
        losses = estimate_loss(model)
        print(f"\nEpoch: {epoch} | Training Loss: {losses['train']: .4f} | Validation Loss: {losses['valid']: .4f}")

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)

    # Performing backpropagation and gradient descent
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

### Making Predictions
xb, yb = get_batch()

for i in range(BATCH_SIZE):
    x_first_sample, y_first_sample = xb[i], yb[i]

    print(decoder(x_first_sample.tolist()).split('}')[0][1:])
    print(model.predict(x_first_sample))
    print(y_first_sample.tolist())
    print()
