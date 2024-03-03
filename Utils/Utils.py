from torch.utils.tensorboard import SummaryWriter
import torch
from torch import save
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from os import remove
from os.path import join
from datetime import datetime


def default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(model, saved_model_path: str, if_exists_stop=False):
    target_path = Path('/'.join(saved_model_path.split('/')[:-1]))
    model_name = saved_model_path.split('/')[-1]

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "Wrong extension: Expecting `.pt` or `.pth`..."
    
    # Creating the directory that the model is going to be saved if not exists
    if not target_path.exists():
        target_path.mkdir(parents=True, exist_ok=True)

    if (target_path / model_name).is_file():
        print(f"[INFO] Model `{model_name}` already exists on `{target_path}`...")
        if if_exists_stop: return
        print(f"[INFO] Deleting `{target_path / model_name}`...")
        remove(target_path / model_name)

    # Saving the Model to path
    print(f"[INFO] Saving Model `{model_name}` to `{target_path}`...")
    save(obj=model.state_dict(), f=target_path/model_name)

    print(f"[INFO] Model Successfully Saved to {target_path / model_name}")

    return target_path / model_name


def create_writer(experiment_name, model_name, extras=None):
    # Loading and formating properly the current time
    timestamp = datetime.now().strftime("%Y-%m-%d")

    # Creating the `log_dir` path
    if extras:
        log_dir = join("runs", timestamp, experiment_name, model_name, extras)
    else:
        log_dir = join("runs", timestamp, experiment_name, model_name)

    print(f"[INFO] SummaryWriter created, saving to: {log_dir}...")
    
    return SummaryWriter(log_dir=log_dir)


@torch.inference_mode()
def plot_image_predictions(model, class_names, image_path, transforms):
    """ Expecting Images of (C, W, H) """

    # Converting the image to Tensor
    with Image.open(image_path, "r") as img:
        img_tensor = transforms(img)

    model.eval()
    logits = model(img_tensor.unsqueeze(dim=0).to(next(model.parameters()).device))

    pred_label = class_names[torch.softmax(logits, dim=1).argmax(dim=1)]
    label_prob = torch.softmax(logits, dim=1).max().item() * 100

    plt.imshow(img_tensor.permute(1, 2, 0).cpu())
    plt.title(f"Label: {pred_label} | Probability: {label_prob: .2f}%")
    plt.axis(False)
    plt.show()

    model.train()


def plot_loss(res, fig_size=(6, 4), font_size=11):
    """ `res` are comming from the `fit` method: Models.TextUtils"""

    plt.figure(figsize=fig_size)
    
    plt.plot(range(res["model_epochs"]), res["model_valid_loss"], c='g', label="Valid Loss")
    plt.plot(range(res["model_epochs"]), res["model_train_loss"], c='b', label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Loss Curves: {res['model_name']}", fontsize=14)
    plt.legend(fontsize=font_size)
    plt.show()
