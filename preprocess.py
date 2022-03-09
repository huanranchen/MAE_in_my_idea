
from PIL import Image
import os
from torchvision.transforms import ToTensor, ToPILImage
import torch
import numpy as np




def resize(target_size=(96,96), path='wait_train.jpg', ):
    BASE_DIR = os.getcwd()

    img_raw = Image.open(os.path.join(BASE_DIR, path))
    h, w = img_raw.height, img_raw.width
    ratio = h / w
    print(f"image hxw: {h} x {w} mode: {img_raw.mode}")

    img = img_raw.resize(target_size)
    rh, rw = img.height, img.width
    print(f'resized image hxw: {rh} x {rw} mode: {img.mode}')
    img.save(os.path.join(BASE_DIR, 'train.jpg'))

def read_one_file_into_memory(path = 'train.jpg'):
    BASE_DIR = os.getcwd()

    img_raw = Image.open(os.path.join(BASE_DIR, path))
    img = np.asarray(img_raw)

    x = torch.from_numpy(img)
    x = x.unsqueeze(0).float()

    return x

def read_batch_into_memory(batch_size = 100):
    pass

def save_img(x,name):
    '''
    :param x: tensor
    :return: save it
    只能处理一个！！！一个！！！
    '''
    x = x.squeeze(0)

    BASE_DIR = os.getcwd()

    x = x.numpy()

    x = Image.fromarray(np.uint8(x))


    x.save(os.path.join(BASE_DIR, name+'.jpg'))