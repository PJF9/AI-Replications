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

