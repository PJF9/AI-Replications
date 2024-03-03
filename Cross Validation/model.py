from ...Utils.ModelUtils import ModelUtils
import torch
from torch import nn
from torch.nn import functional as F


class TransformerEncoder(ModelUtils):
    ### Defining the Components of the Encoder
    class _SelfAttention(nn.Module):
        def __init__(self, embed_size, head_size, dropout):
            super().__init__()
    
            self.query = nn.Linear(embed_size, head_size, bias=False)
            self.key = nn.Linear(embed_size, head_size, bias=False)
            self.value = nn.Linear(embed_size, head_size, bias=False)
    
            self.dropout = nn.Dropout(dropout)
    
        def forward(self, x):
            _, _, C = x.shape # B: BATCH_SIZE, T: max_sentence_length, C: head_size
    
            q = self.query(x) # (B, T, C)
            k = self.key(x)   # (B, T, C)
    
            wei = q @ k.transpose(-2, -1) * (C**-0.5) # (B, T, C) @ (B, C, T) --> (B, T, T)
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
    
            v = self.value(x) # (B, T, C)
    
            out = wei @ v # (B, T, T) @ (B, T, C) --> (B, T, C)
    
            return out
    

    class _MultiHeadAttention(nn.Module):
        def __init__(self, embed_size, head_size, num_att_heads, dropout):
            super().__init__()
    
            self.heads = nn.ModuleList([TransformerEncoder._SelfAttention(embed_size, head_size, dropout) for _ in range(num_att_heads)])
            self.proj = nn.Linear(embed_size, embed_size, dropout)
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

    
    class _Block(nn.Module):  # combining Masked Multi-Head Attention and one Feed-Forward layer
        def __init__(self, embed_size, num_heads, scale_embeds, dropout):
            super().__init__()

            head_size = embed_size // num_heads # because the result of the Masked Multi-Head layer we want to have shape: (B, T, EMBED_SIZE)
            self.multi_att_m = TransformerEncoder._MultiHeadAttention(embed_size, head_size, num_heads, dropout)
            self.ffwd = TransformerEncoder._FeedForward(embed_size, scale_embeds, dropout)
            self.ln1 = nn.LayerNorm(embed_size)
            self.ln2 = nn.LayerNorm(embed_size)

        def forward(self, x): 
            x = x + self.multi_att_m(self.ln1(x)) # (B, T, EMBED_SIZE)
            x = x + self.ffwd(self.ln2(x))        # (B, T, EMBED_SIZE)
    
            return x
            
    ### Initializing the Encoder from its components
    def __init__(self, vocab_size, max_length, embed_size, num_layers, num_heads, scale_embeds, dropout, output_size, device):
        super().__init__()
        
        self.device = device
        self.max_length = max_length

        self.embedding_table = nn.Embedding(vocab_size, embed_size)
        self.position_embedding_table = nn.Embedding(max_length, embed_size)

        self.block = nn.Sequential(*[TransformerEncoder._Block(embed_size, num_heads, scale_embeds, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_size)
        self.linear_head = nn.Linear(embed_size, output_size)

    def forward(self, idx, targets=None):
        token_embeddings = self.embedding_table(idx)                                                          # (B, T, EMBED_SIZE)
        position_embeddings = self.position_embedding_table(torch.arange(self.max_length, device=self.device)) # (T, EMBED_SIZE)

        x = token_embeddings + position_embeddings # (B, T, EMBED_SIZE)

        x = self.block(x) # (B, T, EMBED_SIZE)
        x = self.ln_f(x)  # (B, T, EMBED_SIZE)

        logits = self.linear_head(x) # (B, T, output_size)

        # Condition to seperate training and generating phase
        loss = F.cross_entropy(logits[:, 0, :], targets.type(torch.LongTensor).to(self.device)) if targets is not None else None

        return logits, loss
