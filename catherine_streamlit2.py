#https://github.com/jalalmansoori19/Classfiying-Iris-flower-species/blob/master/requirements.txt
'''>python -m venv env
   > env\scripts\activate
   >python   .py
   >streamlit run .py
'''




#pip install -r requirements.tx   OK
#pip install torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html avec 
    #torch===1.6.0       OK
    #torchvision===0.7.0 OK
    #CUDA 10.2           OK  
  
import argparse
import streamlit as st
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchnet.meter import AverageValueMeter
import torchvision.utils as vutils

parser = {
    'data_path': '../data/ut-zap50k/Shoes/Sneakers_and_athletic_shoes/',
    'epochs': 50, #100
    'batch_size': 64,
    'lr': 0.0002,
    'image_size': 136,
    'scale_size': 64,
    'z_dim': 100,
    'G_features': 64,
    'D_features': 64,
    'image_channels': 3, #nc - number of color channels in the input images. For color images this is 3
    'beta1': 0.5,
    'cuda': True,
    'seed': 7,
    'workers': 2,
    'results': './resultsDCGAN4_0605/'
}

args = argparse.Namespace(**parser)
args.image_results = args.results + 'images/'
args.loss_results = args.results + 'loss/'
args.cuda = args.cuda and torch.cuda.is_available()

class _netG(nn.Module):
    def __init__(self):
        super(_netG, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(args.z_dim, args.G_features * 8,
                               4, 1, 0, bias=False),
            nn.BatchNorm2d(args.G_features * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(args.G_features * 8, args.G_features * 4,
                               4, 2, 1, bias=False),
            nn.BatchNorm2d(args.G_features * 4),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(args.G_features * 4, args.G_features * 2,
                               4, 2, 1, bias=False),
            nn.BatchNorm2d(args.G_features * 2),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(args.G_features * 2, args.G_features,
                               4, 2, 1, bias=False),
            nn.BatchNorm2d(args.G_features),
            nn.ReLU(True),
            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d(args.G_features, args.image_channels,
                               4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 32 x 32
        )
        self._initialize_weights()
        
    # custom weight initialization
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                init.normal(m.weight, mean=0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal(m.weight, mean=1, std=0.02)
                init.constant(m.bias, 0)
      
    def forward(self, x):
        return self.main(x)
    
netG = _netG()

#netG = nn.DataParallel(netG)
#netG = nn.DataParallel(netG)


netG.load_state_dict(torch.load(r'C:\Users\cle35\Documents\SIMPLON\CHEF_OEUVRE\chef_d_oeuvre_IA_DATA\WEB_APP\poids\modelG.pth', map_location=torch.device('cpu')))

st.title('Assitant designer')


def generate_image(model):
    z=torch.randn(args.batch_size, args.z_dim, 1, 1)
    if args.cuda:
        z = z.cuda()
        
    with torch.no_grad():
        fake = model(Variable(z, volatile=True)) 
    

        save_image(fake.data.cpu(), os.path.join(r'C:\Users\cle35\Documents\SIMPLON\CHEF_OEUVRE\chef_d_oeuvre_IA_DATA\WEB_APP',"fake_epoch.png"), normalize=True)
        #im = Image.open("scalebar.png")
        #width, height = im.size
        #print(width, height)# NCHW => NHWC


#st.write(z, 'tensor', torch.randn(args.batch_size, args.z_dim, 1, 1))

#generate_image(netG)




if st.button('Générer image'):
    
    images_saved = generate_image(netG)
    setosa= Image.open('fake_epoch.png')
    

    #plt.figure(figsize=(8,8))
    #plt.axis("off")
    #plt.title("Generated Image After Loading weights")
    #fake = plt.imshow(np.transpose(vutils.make_grid(fake).cpu(),(1,2,0))) 
    
   
    st.image(setosa)

