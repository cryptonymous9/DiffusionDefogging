import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as T
from torchvision.transforms import ToTensor

import pytorch_lightning as pl
import kornia.augmentation as KA

import os
import numpy as np
import imageio
from skimage import io
import matplotlib as mpl
import matplotlib.pyplot as plt

from dataset import ImageDataset
from src import *
from eval import get_criterion

# Settings
CROP_SIZE=256


# Arguments

ROOT_DIR = 'consistent'
TRAIN_TEST_SPLIT=0.8
torch.set_float32_matmul_precision('high')
CHECKPOINT = "./checkpoint/checkpoint-epoch=80-val_loss=0.02582.ckpt"

# Data
def get_dataset(train_test_split):
    T=[
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

train_ds, test_ds = get_dataset(TRAIN_TEST_SPLIT)
model=LatentDiffusionConditional(train_dataset = train_ds, valid_dataset=test_ds, lr=1e-4, batch_size=8)

checkpoint = torch.load(CHECKPOINT)
model.load_state_dict(checkpoint['state_dict'])

class Inference:
    def __init__(self, model, train_ds, test_ds):
        self.model = model
        self.model.eval()
        
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.dat = None
    def get_prediction(self, dat_type, idx, runs, verbose):
        with torch.no_grad():
            #input_,output_ = self.test_ds[idx] if dat_type=='test' else self.train_ds[idx]
            if dat_type == 'test':
                input_,output_= self.test_ds[idx]
            else:
                input_,output_= self.train_ds[idx]
            batch_input=torch.stack(runs*[input_],0)
            self.model.cuda()
            out=self.model(batch_input, verbose=True)
        return input_, output_, out
    
    def evaluate(self, dat_type, criterion):
        dat = self.test_ds if dat_type=='test' else self.train_ds 
        if criterion=='benchmark':
            cc = ['MSE', 'PSNR']
            loss=dict()
            for each in cc:
                loss[each]=0
            eval_ = [get_criterion(each) for each in cc]
        else:
            cc = [criterion]
            loss = {criterion:0}
            eval_ = [get_criterion(criterion)]
        
        with torch.no_grad():
            self.model.cuda()
            for idx, batch_imgs in enumerate(dat):
                input_, output_ = batch_imgs
                out=self.model(input_.cuda(), verbose=True)
                for i in range(len(cc)):
                    func_cr = eval_[i]
                    # print(out.shape, output_.shape)
                    loss[cc[i]]+=func_cr(out.detach().cpu()*255, output_*255)
            loss = {k: v / len(dat) for k, v in loss.items()}
        return loss
    
    def show_predictions(self, dat_type, nsamples, runs, filename):
        rows, columns =  nsamples, runs+2
        fig = plt.figure()
        f, axarr = plt.subplots(rows,columns)         
        indexes = np.random.randint(1,len(self.test_ds), nsamples)        
        for i in range(rows):
            idx = indexes[i]
            input_, output_, out = self.get_prediction(dat_type, idx, runs, verbose=False)
            for j in range(columns):
                if j==0:
                    input_ = retrieve_img(input_.permute(1,2,0))
                    axarr[i,j].imshow(input_)
                    axarr[i,j].set_title('Input') if i==0 else None
                elif j==columns-1:
                    output_ = retrieve_img(output_.permute(1,2,0))
                    axarr[i,j].imshow(output_)
                    axarr[i,j].set_title('GroundTruth') if i==0 else None            
                else:
                    pred = out[j-2].detach().cpu()
                    pred = retrieve_img(pred.permute(1,2,0))
                    axarr[i,j].set_title(f'DD #{j}') if i==0 else None            
                    axarr[i,j].imshow(pred)
                axarr[i,j].axis('off')
        f.savefig(filename,dpi=800)        
        fig.show()   

# defog = Inference(model,train_ds, test_ds)
# defog.show_predictions(dat_type='test', nsamples=5,runs=4)


# Evaluations

BATCH_SIZE = 32
#train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
train_loader = None
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

defog = Inference(model,train_loader, test_loader)
loss = defog.evaluate(dat_type='test', criterion='benchmark')
print(loss)

