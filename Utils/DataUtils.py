import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from random import randint
from pathlib import Path
from PIL import Image
from os import remove, scandir
from shutil import rmtree
import requests
import zipfile


class ImageDataset(Dataset):
    """ `root`: The Path object which contains the path of the dataset on the disk in format: (dataset->classes->images) """
    def __init__(self, root: Path, transform=None):
        self.paths = list(root.glob("*/*"))
        self.transform = transform
        self.classes = [entry.name for entry in list(scandir(root))]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_classes = {i: c for i, c in enumerate(self.classes)}

    def load_image(self, index):
        if (index >= 0) and (index < len(self.paths)):
            return Image.open(self.paths[index])
        raise IndexError

    def __getitem__(self, index):
            if self.transform is not None:
                return (self.transform(self.load_image(index)), self.class_to_idx[self.paths[index].parent.stem])
            return (self.load_image(index), self.class_to_idx[self.paths[index].parent.stem])

    def __len__(self):
        return len(self.paths)


class TextDataset(Dataset):
    """ 
    `text_c`: The column of `df` which contains the tokenized and encoded version of the text 
    If `text_c` and `target_c` are not being passed, then assuming first column Label and second Text
    """

    def __init__(self, df, text_c: str=None, target_c: str=None, classes: list=[], to_tensors: bool=False):
        super().__init__()
        
        if (text_c is None) and (target_c is None):
            target_c, text_c = iter(df)
        
        self.samples = [(df[text_c][i], df[target_c][i]) for i in range(len(df))]
        self.classes = classes
        self.to_tensors = to_tensors
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_classes = {i: c for i, c in enumerate(self.classes)}
    
    def __getitem__(self, index):
        if isinstance(index, slice):
            if self.to_tensors:
                return [(torch.tensor(sample[0], dtype=torch.long), sample[1]) for sample in self.samples[index]] # List (Tuple (Tensor, Int) )
            return [(sample[0], sample[1]) for sample in self.samples[index]]                                     # List (Tuple (Type, Int) )
        
        if self.to_tensors:
            return (torch.tensor(self.samples[index][0], dtype=torch.long), self.samples[index][1]) # Tuple (Tensor, Int)
        return (self.samples[index][0], self.samples[index][1])                                     # Tuple (Type, Int)

    def __len__(self):
        return len(self.samples)


def unzipping_dataset(source: str, dest: str, stops: bool=False, del_zip: bool=True, sep='\\'):
    ds_name = source.split(sep)[-1].split('.')[0] # the name of the dataset
    zip_path = Path(source)
    dest_path = Path(dest + ds_name)

    if dest_path.is_dir():
        print(f"[INFO] `{dest_path}` already exists...")
        if stops: return
        print(f"[INFO] Deleting `{dest_path}`..")
        rmtree(dest_path)

    # Creating the dataset directory
    print(f"[INFO] Creating `{dest_path}`...")
    dest_path.mkdir(parents=True, exist_ok=True)

    # Unziping the Dataset
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        print(f"[INFO] Unzipping dataset `{zip_path}` to `{dest_path}`...")
        zip_ref.extractall(dest_path)
    
    # Deleting the zip file
    if del_zip:
        print(f"[INFO] Deleting `{zip_path}`...")
        remove(zip_path)

    print(f"[INFO] Dataset succesfully downloaded to `{dest_path}`...")

    return dest_path


def download_dataset_zip(source: str, dest: str, stops: bool=False, del_zip: bool=True):
    """ For github replace `blob` with `raw` to able to downlad """

    # Creating the Path object and getting the name of the `zip` file
    ds_name = source.split('/')[-1]
    dataset_path = Path(dest) / ds_name.split('.')[0]

    # If Path object exists then give the option to delete it or stop the execution
    if dataset_path.is_dir():
        print(f"[INFO] `{dataset_path}` already exists...")
        if stops: return
        print(f"[INFO] Deleting `{dataset_path}`..")
        rmtree(dataset_path)

    # Creating the dataset directory
    print(f"[INFO] Creating `{dataset_path}`...")
    dataset_path.mkdir(parents=True, exist_ok=True)

    # Downloading the zip file on the Path object
    with open(dataset_path / ds_name, "wb") as f:
        print(f"[INFO] Downloading dataset: {source} to `{dataset_path}`...")
        req = requests.get(source)
        f.write(req.content)

    # Extracting the zip file to destination
    with zipfile.ZipFile(dataset_path / ds_name, "r") as zip_ref:
        print(f"[INFO] Unzipping dataset `{dataset_path / ds_name}` to `{dataset_path}`...")
        zip_ref.extractall(dataset_path)
    
    # Deleting the zip file
    if del_zip:
        print(f"[INFO] Deleting `{dataset_path / ds_name}`...")
        remove(dataset_path / ds_name)

    print(f"[INFO] Dataset succesfully downloaded to `{dataset_path}`..")

    return dataset_path


def plot_before_and_after(img_path_l: list, transforms):
    """Expects Images of (C, W, H) tensors """
    plt.figure(figsize=(5, 3*len(img_path_l)))

    i = 0
    for image_path in img_path_l:
        with Image.open(image_path) as img:
            # Plotting the original image
            i += 1
            plt.subplot(len(img_path_l), 2, i)
            plt.imshow(img)
            plt.title(f"Original Shape:\n({img.size[0]}, {img.size[1]}, 3)")
            plt.axis(False)

            # Plotting the transformed image
            i += 1
            transformed_img = transforms(img)
            plt.subplot(len(img_path_l), 2, i)
            plt.imshow(transformed_img.permute(1, 2, 0))
            plt.title(f"Transformed Shape:\n({transformed_img.size(dim=0)}, {transformed_img.size(dim=1)}, {transformed_img.size(dim=2)})")
            plt.axis(False)

            # Adding some padding for better display
            plt.subplots_adjust(left=0.2, right=0.9)
    plt.show()


@torch.inference_mode()
def pred_grid(model, ds, nsamples, fsize, pad, lsize, cmap=None):
    """ Expects Pytorch Dataset with images of (C, W, H) tensors"""
    model.eval()

    device = next(model.parameters()).device
    fig = plt.figure(figsize=(fsize, fsize))

    for i in range(1, nsamples*nsamples + 1):
        random_i = torch.randint(len(ds), size=(1,)).item()
        
        # The Features and the Label
        img, label = ds[random_i]
        img = img.to(device)

        # Model's Predictions
        model_logits = model(img.unsqueeze(dim=0))
        model_label = torch.softmax(model_logits, dim=1).argmax()

        # Plotting the Grid
        fig.add_subplot(nsamples, nsamples, i)
        fig.tight_layout(pad=pad)

        if cmap is not None:
            plt.imshow(img.cpu().squeeze(), cmap="gray")
        else:
            plt.imshow(img.cpu().permute(1, 2, 0))

        # Change Colour Depending if the Prediction is Correct or not
        if (label == model_label):
            plt.title(f"Predicted: {ds.classes[model_label]}\nLabel: {ds.classes[label]}", c='g', fontsize=lsize)
        else:
            plt.title(f"Predicted: {ds.classes[model_label]}\nLabel: {ds.classes[label]}", c='r', fontsize=lsize)
        plt.axis(False)
        plt.show()

    model.train()


def cross_validation(ds, valid_prop, batch_size, num_workers=2, pin_memory=False):
    valid_size = int(len(ds) * valid_prop)
    ridx = randint(0, len(ds) - valid_size)

    return (DataLoader(ds[ridx: ridx + valid_size], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
            DataLoader(ds[:ridx] + ds[ridx + valid_size:], batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory))
