import torch
from torch import nn
from torch.nn import functional as F

from Decoder import TransformerDecoder


### Splitting the Dataset into Batches
def get_batch(split):
    data = train_data if split == "train" else valid_data

    # Creating random indexes for computing the batches
    idx_xb = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))

    x = torch.stack([data[i:BLOCK_SIZE+i] for i in idx_xb])
    y = torch.stack([data[i+1:BLOCK_SIZE+i+1] for i in idx_xb])

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


### Hyperparameters
BATCH_SIZE = 64         # the size of sample (in our case character tokens) to create batches 
BLOCK_SIZE = 256        # the context length (the tokens that each batch will contain): using Attention those tokens are able to communicate
EMBED_SIZE = 384        # the size of the embeddings for each token
EPOCHS = 1_000          # the number of times we iterate all the batches in training
LR = 3e-4               # the learning rate of the optimizer: in our case AdamW
EVAL_INTERVAL = 400     # every after those epochs we are evaluating the loss of the model
EVAL_ITERS = 200        # for how many batches we should calculate the loss when evaluating the model
NUM_HEADS = 6           # the number of Masked Self-Attention layers
NUM_LAYERS = 6          # the number of Block Layer (each Block contains one Mulit-Head Attention and one Feed-Forward layer)
DROPOUT = 0.2           # the dropout we are setting in the layers
SCALE_EMBEDS = 4        # how many times we want to scale the embedding dimension in the Feed-Froward layer

### Setting Devide Agnostic Code
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Loading the Dataset
with open("Hamlet.txt", "r") as f:
    text_ds = f.read()

### Creating Vocabulary and Encoder/Decoder Functions
vocab = sorted(list(set(text_ds)))
vocab_size = len(vocab)

char_to_int = {ch: i for i, ch in enumerate(vocab)}
int_to_char = {i: ch for i, ch in enumerate(vocab)}
encoder = lambda s: [char_to_int[c] for c in s]
decoder = lambda l: "".join([int_to_char[i] for i in l])

### Tokenizing the Dataset
tokenized_ds = torch.tensor(encoder(text_ds), dtype=torch.long, device=device) # `torch.int64`

### Splitting the Dataset into Training and Validation Sets
train_size = int(len(tokenized_ds) * 0.9)
train_data, valid_data = tokenized_ds[:train_size], tokenized_ds[train_size:]


### Initializing the Model
model = TransformerDecoder(
    vocab_size = vocab_size,
    embed_size = EMBED_SIZE,
    block_size = BLOCK_SIZE,
    batch_size = BATCH_SIZE,
    num_layers = NUM_LAYERS,
    num_heads = NUM_HEADS,
    scale_embeds = SCALE_EMBEDS,
    dropout = DROPOUT,
    device = device
).to(device)


### Setting Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


### Training the Model
print("\nStarting Training:")
for epoch in range(1, EPOCHS+1):
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


### Generating Text (exactly 1_000 characters)
idx = torch.zeros((1, 1), dtype=torch.long, device=device)

with open("test.txt", "w") as f:
    f.write(decoder(model.generate(idx, max_new_tokens=1_000)[0].tolist()))
