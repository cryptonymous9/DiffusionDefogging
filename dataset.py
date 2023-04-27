import os
import imageio
from skimage import io

import glob
import pathlib
from PIL import Image 

import torch
from torch.utils.data import Dataset

import kornia
from kornia.utils import image_to_tensor
import kornia.augmentation as KA

DIR_in = './data/leftImg8bit_trainextra_foggy/leftImg8bit_foggy/train_extra/*/*.png'
DIR_out = './data/leftImg8bit_trainextra/leftImg8bit/train_extra/*/*.png'

DIR_f = "/home/nidhin/scratch/defogging/foggy/"
DIR_nf = "/home/nidhin/scratch/defogging/non_foggy/"

SIZE = 256



class ImageDataset(Dataset):
    """Dataset returning images in a folder."""

    def __init__(self,
                 root_dir='consistent',
                 transforms=None,
                 return_pair=False):
        if root_dir == 'consistent':
            self.foggy = DIR_f
            self.non_foggy = DIR_nf
        
        self.transforms = transforms
        self.return_pair=return_pair
        
        if self.transforms is not None:
            data_keys=2*['input']

            self.input_T=KA.container.AugmentationSequential(
                *self.transforms,
                data_keys=data_keys,
                same_on_batch=False
            )
        
        
        self.files_foggy = os.listdir(DIR_f)
        self.files_nfoggy = os.listdir(DIR_nf)
        
        self.files_foggy.sort()
        self.files_nfoggy.sort()
        
        #supported_formats=['png']
        #self.files_foggy=[el for el in os.listdir(self.foggy) if el.split('.')[-1] in supported_formats]
        #self.files_nfoggy=[el for el in os.listdir(self.non_foggy) if el.split('.')[-1] in supported_formats]

    def __len__(self):
        return len(self.files_foggy)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()            

        #img_name = os.path.join(self.root_dir, self.files[idx])
        #image = image_to_tensor(io.imread(img_name))/255
        assert self.files_foggy[idx][:-20] == self.files_nfoggy[idx][:-4]
        foggy_filname = os.path.join(self.foggy, self.files_foggy[idx])
        nfoggy_filname = os.path.join(self.non_foggy, self.files_nfoggy[idx])

        foggy_img = image_to_tensor(io.imread(foggy_filname))/255
        nfoggy_img = image_to_tensor(io.imread(nfoggy_filname))/255


        if self.transforms is not None:
                out = self.input_T(foggy_img, nfoggy_img)
                image=out[0][0]
                image2=out[1][0]

        if self.return_pair:
            return image,image2

        

class Preprocess():
    """ Reduce resolution, move data to scratch space"""
    def __init__(self, DIR_in, DIR_out):
        self.DIR_in = DIR_in
        self.DIR_out = DIR_out
        
    def save_downsample(self, imgs_path, save_dir, size, quality, idx=0):
        WIDTH = 2*size
        HEIGHT = size
        new_size = (WIDTH, HEIGHT)
        n = len(imgs_path)
        for i, file in enumerate(imgs_path):
            pil_image = Image.open(file)
            file_name = pathlib.PurePath(file).parts[-1]
            pil_image = pil_image.resize(new_size, resample=Image.Resampling.BICUBIC)
            pil_image.save(os.path.join(save_dir, file_name), quality=quality, format='png')
            if i%100==0:
                print(f'{i}/{n}',end='\r')
            
    def run_downsample(self, save_DIR_f, save_DIR_nf, size):
        quality = 1
        foggy_img_path = [img for img in glob.glob(self.DIR_in)]
        nfoggy_img_path = [img for img in glob.glob(self.DIR_out)] 
        self.save_downsample(foggy_img_path, save_DIR_f, size, quality, idx=0)
        self.save_downsample(nfoggy_img_path, save_DIR_nf, size, quality, idx=0)
        return None
    
    
    
if __name__ == "__main__":
    pp = Preprocess(DIR_in, DIR_out)
    pp.run_downsample(DIR_f, DIR_nf, size=SIZE)
    print("Downsampling successfull!")