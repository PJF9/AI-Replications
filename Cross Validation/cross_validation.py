import torch
from math import ceil
from random import randint


class TorchLoader: # expects: List ( Tuple (Tensor, Int) )
    def __init__(self, ds, batch_size, shuffle):
        self.batch_size = batch_size
        self.shuffle = shuffle

        self._dsx = [s[0].tolist() for s in ds] # contains the x-values (inputs) of the dataset | `tolist()` converts the x-tensor into Python List | List (List (Int) )
        self._dsy = [s[1] for s in ds]          # contains the y-values (targets) of the dataset | List (Int)

        if shuffle:
            self._temp_dsx = self._dsx.copy() 
            self._temp_dsy = self._dsy.copy()

    def __iter__(self):
        if self.shuffle:
            # Iterating over the number of batches that the dataset is going to bet split
            for _ in range(len(self._dsx) // self.batch_size):
                
                # This random index gives the index of the first sample for the batch
                ridx = randint(0, len(self._temp_dsx) - self.batch_size)

                yield (torch.tensor(self._temp_dsx[ridx: ridx + self.batch_size], dtype=torch.long), torch.tensor(self._temp_dsy[ridx: ridx + self.batch_size], dtype=torch.long))

                # Removing the already `yield`ed batch from the dataset
                self._temp_dsx = self._temp_dsx[:ridx] + self._temp_dsx[ridx + self.batch_size:]
                self._temp_dsy = self._temp_dsy[:ridx] + self._temp_dsy[ridx + self.batch_size:]

            # Returning the last batch, which is not going to contain `batch_size` samples
            if len(self._temp_dsx) > 0:
                yield (torch.tensor(self._temp_dsx, dtype=torch.long), torch.tensor(self._temp_dsy, dtype=torch.long))

            # If we try to iterate again over the loader without those two lines, no samples are going to be returned
            self._temp_dsx = self._dsx.copy()
            self._temp_dsy = self._dsy.copy()

        else:
            j = 0
            for _ in range(ceil(len(self._dsx) / self.batch_size)):
                yield (torch.tensor(self._dsx[j: j + self.batch_size], dtype=torch.long), torch.tensor(self._dsy[j: j + self.batch_size], dtype=torch.long))
                j += self.batch_size

    def __len__(self):
        return ceil(len(self._dsx) / self.batch_size)



def cross_validation(ds, valid_prop, batch_size):
    valid_size = int(len(ds) * valid_prop)
    ridx = randint(0, len(ds) - valid_size)

    return (TorchLoader(ds[ridx: ridx + valid_size], batch_size=batch_size, shuffle=False), TorchLoader(ds[:ridx] + ds[ridx + valid_size:], batch_size=batch_size, shuffle=True))
