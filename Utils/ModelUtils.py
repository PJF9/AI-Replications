import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
from timeit import default_timer as timer
from tqdm import tqdm


class ModelUtils:
    """ Expects model with __call__(x, y) that returns first logits and then loss """
    """ Also for NLP Tasks expect that the logits have already being pooled """

    def __init__(self, device, nclasses):
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
