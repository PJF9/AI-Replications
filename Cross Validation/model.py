import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
from timeit import default_timer as timer
from tqdm import tqdm
from collections import defaultdict
import json



class WordPiece:
    """
    Its O(n) with respect both the `ntokens` and the total documents of `corpus`.
    """
    def __init__(self, corpus=None, ntokens=30_000, cleaning=None):
        if (corpus is not None):
            # Cleaning Corpus
            if (cleaning is not None):
                corpus = [cleaning(text) for text in corpus]

            # Calculating the frequencies of each word (global statistics)
            self._word_freqs = defaultdict(lambda : 0)
            for text in corpus:
                for word in text.split():
                    self._word_freqs[word] += 1

        self._cleaning = cleaning if (cleaning is not None) else lambda text: text
        self._ntokens = ntokens
        self.special_t = ["[CLS]", "[UNK]", "[PAD]", "[SEP]"]
        self.vocab_l = []
        self.vocab_d = {}
        self.ivocab_d = {}
        self.vocab_size = 0
    

    def __calc_pair_scores(self, splits):
        token_freqs = defaultdict(lambda: 0)  # Capturing the global corpus statistics for `freq of firsy element` and `freq of second element`
        pair_freqs = defaultdict(lambda: 0)   # Capturing the statistics for `freq of pair`: How many times this pair appears in the corpus

        # Iterate over all words of the corpus
        for word, freq in self._word_freqs.items():
            split = splits[word]

            # If a word contains only 1 letter
            if len(split) == 1:
                token_freqs[split[0]] += freq
                continue

            for i in range(len(split) - 1):
                token_freqs[split[i]] += freq
                pair_freqs[(split[i], split[i+1])] += freq

            # Adding the final token that the for-loop is not processing
            token_freqs[split[-1]] += freq

        # Returning the scores, calculated from the formula above
        return {pair: pair_freq / (token_freqs[pair[0]] * token_freqs[pair[1]]) for pair, pair_freq in pair_freqs.items()}


    @staticmethod
    def __highest_score(pair_scores):
        max_pair = ('', '')
        max_freq = 0
        for pair, freq in pair_scores.items():
            if freq > max_freq:
                max_freq = freq
                max_pair = pair
        
        return max_pair, max_freq
    
    @staticmethod
    def __merge_pair(pair_tuple, splits):
        # Iterating over the word-tokens defaultdict
        for word in splits.keys():
            split = splits[word] # contains the tokens of the word

            if len(split) == 1:
                continue
            
            # Iterating until we find the pair in the tokenize representation of the word
            i = 0
            while i < len(split) - 1:
                if (split[i] == pair_tuple[0]) and (split[i+1] == pair_tuple[1]):
                    merge = pair_tuple[0] + pair_tuple[1][2:] if pair_tuple[1].startswith("##") else pair_tuple[0] + pair_tuple[1]
                    split = split[:i] + [merge] + split[i+2:]
                else:
                    i += 1

            splits[word] = split
        return splits
    

    def fit(self):
        # Original splits (character tokenization of each word in the corpus)
        splits = {word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)] for word in self._word_freqs.keys()}

        # Creating the basic vocabulary
        vocab_ = set([token for tokens in splits.values() for token in tokens])
        
        new_tokens = []
        for _ in tqdm(range(len(vocab_), self._ntokens), desc="Creating Vocabulary: "):
            ps = self.__calc_pair_scores(splits)   # Calculate the Score of each pair
            bs, _ = self.__highest_score(ps)       # Get the pair with the highest score
            splits = self.__merge_pair(bs, splits) # Merge those two tokens

            if bs[1].startswith("##"):
                new_tokens.append(bs[0] + bs[1][2:])
                continue
            new_tokens.append(bs[0] + bs[1])

        # Adding to the the basic vocabulary the new tokens
        self.vocab_l = self.special_t + sorted(list((vocab_ | set(new_tokens))))
        self.vocab_d = {term: i for i, term in enumerate(self.vocab_l)}
        self.ivocab_d = {i: term for i, term in enumerate(self.vocab_l)}
        self.vocab_size = len(self.vocab_l)

    
    def save_vocab(self, path):
        # Saving the Vocabulary Dict into a JSON file
        with open(path, "w") as f:
            json.dump(self.vocab_d, f)

    def load_vocab(self, path):
        # Updating the Vocabulary elements from the JSON file
        with open(path, "r") as f:
            self.vocab_d = json.loads(f.read())
        self.vocab_l = list(self.vocab_d.keys())
        self.ivocab_d = {i: token for i, token in enumerate(self.vocab_l)}
        self.vocab_size = len(self.vocab_l)


    def __tokenize_word(self, word):
        tokens = []

        # Iterating over the entire word starting from the end
        while len(word) > 0:
            i = len(word)
            # Trying to find the bigest sub-word that exists on our vocabulary
            while (i > 0) and word[:i] not in self.vocab_l:
                i -= 1

            # If a sub-word does not exist on the vocabulary
            if i == 0:
                tokens.append("[UNK]")
                return tokens          # keeping some information about the word
            
            # The first sub-word is not going to contain `##`
            tokens.append(word[:i])
            word = word[i:]

            # All the other sub-words are going to contain `##`
            if len(word) > 0:
                word = f"##{word}"

        return tokens
    
    def __decode_word(self, idx):
        to_tokens = [self.ivocab_d[i] for i in idx]

        return ''.join([token[2:] if token.startswith("##") else token for token in to_tokens])


    def tokenize(self, text, npad=0):
        t_text = []
        for word in self._cleaning(text).split():
            for token in self.__tokenize_word(word):
                t_text.append(token)
            t_text.append("[SEP]")

        for _ in range(npad - len(t_text)):
            t_text.append("[PAD]")

        return t_text

    def encode(self, text, npad=0):
        return [self.vocab_d[token] for token in self.tokenize(text, npad=npad)]

    def decode(self, idx):
        text = ""
        i, j = 0, 0
        while i < len(idx) and idx[i] != self.vocab_d["[PAD]"]:
            if idx[i] == self.vocab_d["[SEP]"]:
                text += self.__decode_word(idx[j: i]) + " "
                j = i + 1
            i += 1

        return text




class ModelUtils(nn.Module):
    """ Expects model with __call__(x, y) that returns first logits and then loss """
    """ Also for NLP Tasks expect that the logits have already being pooled """

    def __init__(self, device, nclasses):
        super().__init__()
        self.device = device
        self.nclasses = nclasses


    def __training_step(self, train_dl, opt):
        losses = torch.zeros(len(train_dl), device=self.device)
        for i, (x_train, y_train) in enumerate(train_dl):
            x_train, y_train = x_train.to(self.device), y_train.to(self.device)

            _, loss = self(x_train, y_train)
            losses[i] = loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()

        return losses.mean().item()
    
    def __training_step_tqdm(self, train_dl, opt):
        losses = torch.zeros(len(train_dl), device=self.device)
        i = 0
        for x_train, y_train in tqdm(train_dl, desc="\tTraining Phase: "):
            x_train, y_train = x_train.to(self.device), y_train.to(self.device)

            _, loss = self(x_train, y_train)
            losses[i] = loss.item()
            i += 1

            opt.zero_grad()
            loss.backward()
            opt.step()

        return losses.mean().item()

    @torch.inference_mode()
    def __validation_step(self, valid_dl):
        self.eval()
        losses = torch.zeros(len(valid_dl), device=self.device)
        for i, (x_train, y_train) in enumerate(valid_dl):
            x_train, y_train = x_train.to(self.device), y_train.to(self.device)

            _, loss = self(x_train, y_train)
            losses[i] = loss.item()

        self.train()
        return losses.mean().item()


    def fit_cv(self, epochs, train_ds, opt, cross_validation, valid_prop, batch_size):
        """ `cross_validation(ds, valid_prop, batch_size)` must return 2 loaders, first validation and then training """
        start_time = timer()
        train_losses, valid_losses = [], []

        t = tqdm(range(epochs), desc="Training Model: ")
        t.set_postfix({"train_loss": "inf", "valid_loss": "inf"})
        for _ in t:
            valid_dl, train_dl = cross_validation(train_ds, valid_prop=valid_prop, batch_size=batch_size)

            train_loss = self.__training_step(train_dl, opt)
            valid_loss = self.__validation_step(valid_dl)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            t.set_postfix({"train_loss": train_loss, "valid_loss": valid_loss})
            t.refresh()

        return {"model_train_loss": train_losses,
                "model_valid_loss": valid_losses,
                "model_name": self.__class__.__name__,
                "model_optimizer": opt.__class__.__name__,
                "model_device": self.device,
                "model_epochs": epochs,
                "model_time": timer() - start_time}

    def fit(self, epochs, train_dl, valid_dl, opt):
        start_time = timer()
        train_losses, valid_losses = [], []

        t = tqdm(range(epochs), desc="Training Model: ")
        t.set_postfix({"train_loss": "inf", "valid_loss": "inf"})
        for _ in t:
            train_loss = self.__training_step(train_dl, opt)
            valid_loss = self.__validation_step(valid_dl)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            t.set_postfix({"train_loss": train_loss, "valid_loss": valid_loss})
            t.refresh()

        return {"model_train_loss": train_losses,
                "model_valid_loss": valid_losses,
                "model_name": self.__class__.__name__,
                "model_optimizer": opt.__class__.__name__,
                "model_device": self.device,
                "model_epochs": epochs,
                "model_time": timer() - start_time}


    def ffit_cv(self, epochs, train_ds, opt, cross_validation, valid_prop, batch_size):
        """ `cross_validation(ds, valid_prop, batch_size)` must return 2 loaders, first validation and then training """
        start_time = timer()
        train_losses, valid_losses, valid_evals = [], [], []

        print("[INFO] Strating Process...")
        for epoch in range(1, epochs + 1):
            print(f"-> Epoch: {epoch}/{epochs}")

            valid_dl, train_dl = cross_validation(train_ds, valid_prop=valid_prop, batch_size=batch_size)

            train_loss = self.__training_step_tqdm(train_dl, opt)
            valid_loss, valid_eval = self.evaluate_classification(valid_dl, _train=True)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            valid_evals.append(valid_eval)

            print()
            print(
                f" \t\tResults:\n",
                f"\t\t--------\n"
                f"\t\tTrain Loss: {train_loss:.4f}\n",
                f"\t\tValid Loss: {valid_loss:.4f}\n"
                f"\t\tValid Accuracy:  {valid_eval['MulticlassAccuracy'] * 100:.2f}%\n",
                f"\t\tValid Precision: {valid_eval['MulticlassPrecision'] * 100:.2f}%\n",
                f"\t\tValid Recall:    {valid_eval['MulticlassRecall'] * 100:.2f}%\n",
                f"\t\tValid F1-Score:  {valid_eval['MulticlassF1Score'] * 100:.2f}%\n"
            )
            print("-" * 85, end="\n\n")

        print("[INFO] Process Completed Successfully...")

        return {"model_train_loss": train_losses,
                "model_valid_loss": valid_losses,
                "model_valid_evals": valid_evals,
                "model_name": self.__class__.__name__,
                "model_optimizer": opt.__class__.__name__,
                "model_device": self.device,
                "model_epochs": epochs,
                "model_time": timer() - start_time}

    def ffit(self, epochs, train_dl, valid_dl, opt):
        start_time = timer()
        train_losses, valid_losses, valid_evals = [], [], []

        print("[INFO] Strating Process...")
        for epoch in range(1, epochs + 1):
            print(f"-> Epoch: {epoch}/{epochs}")

            train_loss = self.__training_step_tqdm(train_dl, opt)
            valid_loss, valid_eval = self.evaluate_classification(valid_dl, _train=True)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            valid_evals.append(valid_eval)

            print()
            print(
                f" \t\tResults:\n",
                f"\t\t--------\n"
                f"\t\tTrain Loss: {train_loss:.4f}\n",
                f"\t\tValid Loss: {valid_loss:.4f}\n"
                f"\t\tValid Accuracy:  {valid_eval['MulticlassAccuracy'] * 100:.2f}%\n",
                f"\t\tValid Precision: {valid_eval['MulticlassPrecision'] * 100:.2f}%\n",
                f"\t\tValid Recall:    {valid_eval['MulticlassRecall'] * 100:.2f}%\n",
                f"\t\tValid F1-Score:  {valid_eval['MulticlassF1Score'] * 100:.2f}%\n"
            )
            print("-" * 85, end="\n\n")

        print("[INFO] Process Completed Successfully...")

        return {"model_train_loss": train_losses,
                "model_valid_loss": valid_losses,
                "model_valid_evals": valid_evals,
                "model_name": self.__class__.__name__,
                "model_optimizer": opt.__class__.__name__,
                "model_device": self.device,
                "model_epochs": epochs,
                "model_time": timer() - start_time}


    @torch.inference_mode()
    def evaluate_classification(self, dl, _train=False):
        """ Evaluating the model over the given Loader """
        self.eval()

        metric_collection = MetricCollection([
            Accuracy(task="multiclass", num_classes=self.nclasses, average="macro"),
            Precision(task="multiclass", num_classes=self.nclasses, average="macro"),
            Recall(task="multiclass", num_classes=self.nclasses, average="macro"),
            F1Score(task="multiclass", num_classes=self.nclasses, average="macro")
        ]).to(self.device)

        losses = torch.zeros(len(dl))
        i = 0
        desc = "\tEvaluating Phase: " if _train else "Evaluating Phase: "
        for xb, yb in tqdm(dl, desc=desc):
            xb, yb = xb.to(self.device), yb.to(self.device)

            logits, loss = self(xb, yb)
            preds = F.softmax(logits, dim=-1)

            metric_collection.update(preds, yb)
            losses[i] = loss.item()
            i += 1

        res = metric_collection.compute()
        
        self.train()
        return losses.mean().item(), res



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
        super().__init__(device, 2)
        
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

    @torch.inference_mode()
    def evaluate(self, dl):
        self.eval()

        device = next(self.parameters()).device
        metric_collection = MetricCollection([
            Accuracy(task="multiclass", num_classes=2, average="macro"),
            Precision(task="multiclass", num_classes=2, average="macro"),
            Recall(task="multiclass", num_classes=2, average="macro"),
            F1Score(task="multiclass", num_classes=2, average="macro")
        ]).to(device)
        losses = torch.zeros(len(dl))

        for i, (xb, yb) in enumerate(dl):
            xb, yb = xb.to(device), yb.to(device)

            logits, loss = self(xb, yb)
            preds = F.softmax(logits, dim=-1)

            metric_collection.update(preds[:, 0, :], yb)
            losses[i] = loss.item()
        
        res = metric_collection.compute()
        
        self.train()
        return losses.mean().item(), res["MulticlassAccuracy"].item(), res["MulticlassPrecision"].item(), res["MulticlassRecall"].item(), res["MulticlassF1Score"].item()