import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as T
from torchvision.transforms import ToTensor

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import kornia.augmentation as KA

import os
import numpy as np
import imageio
from skimage import io
import matplotlib as mpl
import matplotlib.pyplot as plt

from dataset import ImageDataset
from src import *



# Settings
exp_name = 'run_1'
CROP_SIZE=256
CKPT_DIR = './checkpoint'
ROOT_DIR = 'consistent'
TRAIN_TEST_SPLIT = 0.8
TRAIN_TEST_SPLIT=0.8
torch.set_float32_matmul_precision('high')


# Dataset
def get_dataset(train_test_split):
    T=[
        KA.RandomGrayscale(p=0.2),
        KA.RandomHorizontalFlip()
      ]

    ds = ImageDataset(root_dir=ROOT_DIR,
                          transforms=T,
                          return_pair=True
                         )

    train_size = int(train_test_split * len(ds))
    test_size = len(ds) - train_size
    train_ds, test_ds = torch.utils.data.random_split(ds, [train_size, test_size])
    
    return train_ds, test_ds

if __name__ == "__main__":
    train_ds, test_ds = get_dataset(TRAIN_TEST_SPLIT)
    model=LatentDiffusionConditional(train_dataset = train_ds, valid_dataset=test_ds, lr=1e-4, batch_size=8)
    exp_name = 'run_1'
    os.makedirs(os.path.join(CKPT_DIR, exp_name), exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_top_k = 5,
        dirpath=CKPT_DIR,
        filename='checkpoint-{epoch:02d}-{val_loss:.5f}')
    trainer = pl.Trainer(max_steps=2e5, accelerator='gpu', devices = [0], callbacks=[EMA(0.9999), checkpoint_callback])
    trainer.fit(model)
    torch.save(model, os.path.join(CKPT_DIR, exp_name, 'final_ckpt_2e-5.pt'))
    
## testing
#input_,output_=test_ds[0]
#batch_input=torch.stack(4*[input_],0)
#model.cuda()
#out=model(batch_input, verbose=True)