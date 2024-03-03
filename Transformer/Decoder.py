import torch
from torch import nn
from torch.nn import functional as F


class TransformerDecoder(nn.Module):
    ### Defining the Components of the Decoder
    class _SelfAttentionMasked(nn.Module):
        def __init__(self, head_size, embed_size, block_size, dropout):
            super().__init__()

            # Initializing the query, key and value vectors
            self.key = nn.Linear(embed_size, head_size, bias=False)
            self.query = nn.Linear(embed_size, head_size, bias=False)
            self.value = nn.Linear(embed_size, head_size, bias=False)

            # Creating a variable `tril` that is not a parameter
            self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            # Capturing the input dimensions: (B: `batch`, T: `time`, C: `channels`) = (BATCH_SIZE, BLOCK_SIZE, EMBED_SIZE)
            _, _, C = x.shape

            # Calculate key and query
            k = self.key(x)   # (B, T, head_size)
            q = self.query(x) # (B, T, head_size)
            
            # Calculating the weight matrix that captures the connections of the tokens: (B, T, head_size) @ (B, head_size, T) --> (B, T, T)
            wei = q @ k.transpose(-2, -1) * C**-0.5               # 1. Where a single query token needs to give more attention in the key tokens
            wei = wei.masked_fill(self.tril == 0, float("-inf"))  # 2. Masking the attention matrix to hide the future tokens (of a token)
            wei = F.softmax(wei, dim=-1)                          # 3. Converting the attention matrix into probabilities

            wei = self.dropout(wei)

            # Calculate value to determine how much each token want each token is ready to communicate with the other tokens
            v = self.value(x)

            out = wei @ v # (B, T, T) @ (B, T, head_size) --> (B, T, head_size)

            return out


    class _MultiHeadAttentionMasked(nn.Module):
        def __init__(self, head_size, embed_size, num_heads, dropout, block_size):
            super().__init__()

            self.heads = nn.ModuleList([TransformerDecoder._SelfAttentionMasked(head_size, embed_size, block_size, dropout) for _ in range(num_heads)])
            self.proj = nn.Linear(embed_size, embed_size)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            # Contatenate the outputs of each Masked Self-Attention
            out = torch.cat([head(x) for head in self.heads], dim=-1) # (B, T, EMBED_SIZE)
            out = self.dropout(self.proj(out))
            return out


    class _FeedForward(nn.Module):
        def __init__(self, embed_size, scale_embeds, dropout):
            super().__init__()

            self.net = nn.Sequential(
                nn.Linear(embed_size, scale_embeds * embed_size),
                nn.ReLU(),
                nn.Linear(scale_embeds * embed_size, embed_size),
                nn.Dropout(dropout)
            )

        def forward(self, x):
            return self.net(x) # (B, T, EMBED_SIZE)
        

    class _Block(nn.Module): # combining Masked Multi-Head Attention and one Feed-Forward layer
        def __init__(self, embed_size, num_heads, scale_embeds, dropout, block_size):
            super().__init__()

            head_size = embed_size // num_heads # because the result of the Masked Multi-Head layer we want to have shape: (B, T, EMBED_SIZE)
            self.multi_att_m = TransformerDecoder._MultiHeadAttentionMasked(head_size, embed_size, num_heads, dropout, block_size)
            self.ffwd = TransformerDecoder._FeedForward(embed_size, scale_embeds, dropout)
            self.ln1 = nn.LayerNorm(embed_size)
            self.ln2 = nn.LayerNorm(embed_size)

        def forward(self, x): 
            x = x + self.multi_att_m(self.ln1(x)) # (B, T, EMBED_SIZE)
            x = x + self.ffwd(self.ln2(x))        # (B, T, EMBED_SIZE)

            return x


    ## Initializing the Decoder from its components
    def __init__(self, vocab_size, embed_size, block_size, batch_size, num_layers, num_heads, scale_embeds, dropout, device):
        super().__init__()

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

        self.embedding_table = nn.Embedding(vocab_size, embed_size)
        # Self-Attention doesn't take into consideration the position of tokens when computing the attetnion matrix, so we have to
        self.position_embedding_table = nn.Embedding(block_size, embed_size)
        self.block = nn.Sequential(*[TransformerDecoder._Block(embed_size, num_heads, scale_embeds, dropout, block_size) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_size)
        self.linear_head = nn.Linear(embed_size, vocab_size)        

    def forward(self, idx, targets=None):
        token_embeddings = self.embedding_table(idx)                                                 # (B, T, EMBED_SIZE)
        position_embeddings = self.position_embedding_table(torch.arange(idx.shape[1], device=self.device)) # (T, EMBED_SIZE)

        x = token_embeddings + position_embeddings # (B, T, EMBED_SIZE)

        x = self.block(x) # (B, T, EMBED_SIZE)
        x = self.ln_f(x)  # (B, T, EMBED_SIZE)

        logits = self.linear_head(x) # (B, T, vocab_size)
        
        # Condition to seperate training and generating phase
        if targets is None:
            loss = None
        else:
            logits = logits.view(self.batch_size*self.block_size, self.vocab_size)
            targets = targets.view(self.batch_size*self.block_size)

            loss = F.cross_entropy(logits, targets) # expects: (B*T, vocab_size) as logit tensor

        return logits, loss

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            # Due to positional embeddings is only rational to pass T size of character tokens into the model
            idx_cond = idx[:, -self.block_size:] # conditioning the input in order for each generation to consider the last T characters

            logits, _ = self(idx_cond) # (B, T, vocab_size)
    
            # We want the output of the last time step to generate the next
            logits = logits[:, -1, :] # (B, vocab_size)

            probs = F.softmax(logits, dim=1) # (B, vocab_size)

            # For each batch getting the index of the highest probability (using multinomial distribution)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        self.train()
        return idx
