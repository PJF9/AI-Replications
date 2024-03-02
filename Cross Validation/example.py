

### We are going to test the cross validation function on the NLP task spam detection

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

# Getting max text length for padding
max_len = df["Cl_Text"].str.len().max() + 1 # due to [SEP] at the end of each sentence

df["En_Text"] = df["Cl_Text"].apply(lambda x: [w.vocab_d["[CLS]"]] + w.encode(text=x, npad=max_len))

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

ds = FinancialNewsDataset(df, ["ham", "spam"])
classes = ds.classes
class_to_idx = ds.class_to_idx
idx_to_classes = ds.idx_to_classes

# Creating the Model
class ModelUtils(nn.Module):
    def __init__(self):
        super().__init__()
    

    def __training_step(self, train_dl, opt, device):
        losses = torch.zeros(len(train_dl), device=device)
        for i, (x_train, y_train) in enumerate(train_dl):
            x_train, y_train = x_train.to(device), y_train.to(device)

            _, loss = self(x_train, y_train)
            losses[i] = loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()

        return losses.mean().item()


    @torch.inference_mode()
    def __validation_step(self, valid_dl, device):
        self.eval()
        losses = torch.zeros(len(valid_dl), device=device)
        for i, (x_train, y_train) in enumerate(valid_dl):
            x_train, y_train = x_train.to(device), y_train.to(device)

            _, loss = self(x_train, y_train)
            losses[i] = loss.item()

        self.train()
        return losses.mean().item()


    def fit(self, epochs, train_ds, opt):
        start_time = timer()
        device = next(self.parameters()).device
        train_losses, valid_losses = [], []

        t = tqdm(range(1, epochs + 1), desc="Training Model: ")
        t.set_postfix({"train_loss": "inf", "valid_loss": "inf"})
        for _ in t:
            valid_dl, train_dl = cross_validation(train_ds, valid_prop=0.2, batch_size=32)
        
            train_loss = self.__training_step(train_dl, opt, device)
            valid_loss = self.__validation_step(valid_dl, device)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            t.set_postfix({"train_loss": train_loss, "valid_loss": valid_loss})
            t.refresh()

        return {"model_train_loss": train_losses,
            "model_valid_loss": valid_losses,
            "model_name": self.__class__.__name__,
            "model_optimizer": opt.__class__.__name__,
            "model_device": device.type,
            "model_epochs": epochs,
            "model_time": timer() - start_time}


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


