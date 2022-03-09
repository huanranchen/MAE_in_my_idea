import torch
import torch.nn as nn
from ViT import ViT
from MAE import MAE
from preprocess import *
import os


def train():
    pass


def train_only_one_img(num_epoch=100, lr=0.01):
    criterion = nn.MSELoss(reduction='mean')
    encoder = ViT()
    decoder = ViT()
    model = MAE(encoder, decoder)
    if os.path.exists('QingBinLi'):
        model.load_state_dict(torch.load('QingBinLi'))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for i in range(num_epoch):
        optimizer.zero_grad()
        x = read_one_file_into_memory()
        y = x.clone()
        pre = model(x)
        loss = criterion(pre, y)
        print(loss.item())
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), 'QingBinLi')


def visualization():
    encoder = ViT()
    decoder = ViT()
    model = MAE(encoder, decoder)
    if os.path.exists('QingBinLi'):
        model.load_state_dict(torch.load('QingBinLi'))
    with torch.no_grad():
        x = read_one_file_into_memory()
        y = x.clone()
        mask = model.mask(x)

        mask = mask.permute(0, 2, 3, 1)
        save_img(mask, 'masked')
        pre = model(x)
        save_img(pre, 'recover')