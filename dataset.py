
import torch
import os
import pandas as pd
from PIL import Image
import torchvision
import torchvision.transforms as transforms



class Fruit_Dataset(torch.utils.data.Dataset):
  def __init__(self, csv_file, root_dir="",transform=None):
      self.root_dir = root_dir
      self.transform = transform
      self.dataset = pd.read_csv(csv_file)

  def __len__(self):
        return len(self.dataset)

  def __getitem__(self, idx):
      image_path = self.dataset.iloc[idx,0]
      class_name = self.dataset.iloc[idx,1]
      class_id =   self.dataset.iloc[idx,2]
      image = Image.open(image_path).convert('RGB')
      transform = transforms.Compose([
      torchvision.transforms.Resize((256,256)),
      torchvision.transforms.ToTensor(),
      transforms.Normalize(mean=[0.8791, 0.7615, 0.6192],std=[0.1720, 0.2586, 0.3450])
        ])
      image_tensor = transform(image)

      if self.transform:
        image = self.transform(image)

      return image_tensor,class_name,class_id