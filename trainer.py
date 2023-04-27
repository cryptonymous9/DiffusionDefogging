from ast import arg
import os 
import sys
import time
import argparse
import pickle
import numpy as np

import torch
from torch.optim import Adam

need_btch_logs = False
batches_log = 100
ckpt_save = 2
ckpt_dir = './checkpoints'


Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

class Trainer:
    def __init__(self, model, train_iterator, val_iterator, lr):
        self.model = model
        self.train_iterator = train_iterator 
        self.val_iterator = val_iterator
        self.max_step = len(train_iterator)
        
        self.criterion = torch.nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=lr)
    
    def save_ckpt(self, params, epoch, text_='_'):
            torch.save({
                'epoch': epoch, 
                'state_dict': params}, 
                ckpt_dir + '/epoch_' + str(epoch).zfill(3) + text_ + '.pth')
                
    def train_model(self, total_epoch):
        train_loss, test_loss = [], []
        best_loss = np.inf
        for epoch in range(total_epoch):
            avg_train  = 0 
            self.model.train()
            
            for batch_idx, (images, gt) in enumerate(tqdm(self.train_iterator)):
                images = [each_img_.type(Tensor) for each_img_ in images]
                gt = [each_gt_.type(Tensor) for each_gt_ in gt]

                self.optimizer.zero_grad()

                try:
                    out = self.model(images)
                except RuntimeError:
                    print("Runtime error, Batch Id:", batch_idx)
                    return True
                
                loss = self.criterion(out, gt)
                loss.backward()
                self.optimizer.step()

                avg_train+=loss.item()
                if need_btch_logs and (batch_idx % batches_log == 0:
                    print(f"\tEpoch {epoch} [{batch_idx}/{self.max_step}] - Train loss: {loss.item()}")
            
            train_loss.append(avg_train/self.max_step)
            tloss = validate(self.model, self.val_iterator, loss_type="MSE")
            test_loss.append(tloss)

            if epoch % ckpt_save == 0:
                self.save_ckpt(self.model.state_dict(), epoch)
            
            if epoch % args.epoch_log == 0:
                print(f"Epoch {epoch} - Train loss: {train_loss[-1]}, Test loss: {tloss}, Prev Best loss: {best_loss}")
                if best_loss > tloss:
                    best_epoch = epoch
                    best_loss = tloss
                    best_parameters = self.model.state_dict()
                    print("Validation improved.. \n")

        dic_ = {"train_loss":train_loss, "test_loss": test_loss}
        self.save_file(filename= ckpt_dir+'/logs.pkl', data=dic_)
        self.save_ckpt(best_parameters, best_epoch, text_='_best')
