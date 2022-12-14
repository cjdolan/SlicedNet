import pickle
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision.transforms import Normalize
import torchvision.transforms as transforms

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_file(file):
  p_file = unpickle(file)
  meta = unpickle('/content/cifar-100-python/meta')
  p_file[b'data'] = p_file[b'data'].reshape(len(p_file[b'data']),3,32,32).transpose(0,2,3,1)
  label_names = []
  for l in p_file[b'fine_labels']:
    label_names.append(meta[b'fine_label_names'][l])
  coarse_label_names = []
  for l in p_file[b'coarse_labels']:
    coarse_label_names.append(meta[b'coarse_label_names'][l])
  p_file['label_names'] = label_names
  p_file['coarse_label_names'] = coarse_label_names
  return p_file

def prep_data(df, data):
  train_data = {}
  train_data['data'] = []
  train_data['fine_labels'] = []
  train_data['label_names'] = []
  for index, row in tqdm(df.iterrows(), total=df.shape[0], position=0, leave=True):
    train_data['data'].append(data[b'data'][row['image_id']])
    train_data['fine_labels'].append(data[b'fine_labels'][row['image_id']])
    train_data['label_names'].append(data['label_names'][row['image_id']])
  train_data['data'] = np.array(train_data['data'])
  train_data['fine_labels'] = np.array(train_data['fine_labels'])
  train_data['label_names'] = np.array(train_data['label_names'])
  
  return train_data

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

class Cifar100Dataset(Dataset):
  def __init__(self, data, augment):
    self.data = data
    self.augment = augment
    self.data['encoded_labels'] = []
    for label in self.data['fine_labels']:
      temp = [0.0 for i in range(100)]
      temp[label] = 1.0
      self.data['encoded_labels'].append(temp)
    self.data['encoded_labels'] = np.array(self.data['encoded_labels'])

  def __getitem__(self, index):
    image = self.data['data'][index]
    image = image / 255
    label = self.data['fine_labels'][index]
    if self.augment:
      transformer = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
      ])
    else:
      transformer = transforms.Compose([
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
      ])
    image = torch.as_tensor(image).permute(2,0,1)
    image = transformer(image)

    return image, torch.tensor(label)

  def __len__(self):
    return self.data['data'].shape[0]