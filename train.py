import glob
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import Fruit_Dataset
from model import Net



def prepare_dataset():
  train_root_dir = "manishghop-fruits_360-f0a8c45800d3/train"
  test_root_dir = "manishghop-fruits_360-f0a8c45800d3/test"
  validation_root_dir = "manishghop-fruits_360-f0a8c45800d3/validation"

  unique_fruits={}
  counter=0
  for folder in os.listdir(train_root_dir):
    fruit_name = folder
    if fruit_name not in unique_fruits:
      unique_fruits[fruit_name] = counter
      counter+=1

  print(unique_fruits)

  for folder in os.listdir(train_root_dir):
      for image_name in os.listdir(os.path.join(train_root_dir,folder)):
        print(os.path.join(train_root_dir,folder,image_name))


  with open('fruits_train.csv','w+') as f:
    f.write('image_path')
    f.write(',')
    f.write('class_name')
    f.write(',')
    f.write('class_id')
    f.write('\n')

    for folder in os.listdir(train_root_dir):
      for image_name in os.listdir(os.path.join(train_root_dir,folder)):
        f.write(os.path.join(train_root_dir,folder,image_name))
        f.write(',')
        fruit_name = folder
        f.write(fruit_name)
        f.write(',')
        f.write(str(unique_fruits[fruit_name]))
        f.write('\n')

  with open('fruits_validation.csv','w+') as f:
    f.write('image_path')
    f.write(',')
    f.write('class_name')
    f.write(',')
    f.write('class_id')
    f.write('\n')

    for folder in os.listdir(validation_root_dir):
      for image_name in os.listdir(os.path.join(validation_root_dir,folder)):
        fruit_name = folder
        if fruit_name not in unique_fruits:
          continue
        f.write(os.path.join(validation_root_dir,folder,image_name))
        f.write(',')

        f.write(fruit_name)
        f.write(',')
        f.write(str(unique_fruits[fruit_name]))
        f.write('\n')

  with open('fruits_test.csv','w') as f:
    f.write('image_path')
    f.write(',')
    f.write('class_name')
    f.write(',')
    f.write('class_id')
    f.write('\n')

    for folder in os.listdir(test_root_dir):
      for image_name in os.listdir(os.path.join(test_root_dir,folder)):
        f.write(os.path.join(test_root_dir,folder,image_name))
        f.write(',')
        fruit_name = folder
        f.write(fruit_name)
        f.write(',')
        f.write(str(unique_fruits[fruit_name]))
        f.write('\n')


def get_accuracy(prediction,label):
      return (prediction==label).sum().item()/len(label)


def saveModel(net,epoch_number):
      print("Saving model at epoch : {}".format(epoch_number))
      path = "./image_classification_epoch_{0}.pth".format(epoch_number)
      torch.save(net.state_dict(), path)
      return path

def train():
  print("Training started\n")
  batch_size=64

  net = Net()

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

  num_epochs = 10

  best_accuracy = 0.0
  best_epoch_weight_path = ""

  for epoch in range(num_epochs):
    running_loss = 0.0

    for index, data in enumerate(train_loader):

      image,class_name,class_id = data

      optimizer.zero_grad()

      outputs = net(image)

      loss = criterion(outputs,class_id)

      loss.backward()

      optimizer.step()


      running_loss+= loss.item()*image.size(0)
      epoch_loss = running_loss/len(train_dataset)

      if index % 10 == 9:    # print every 10 mini-batches
              total_accuracy = 0.0
              for index,valid_data in enumerate(valid_loader):
                output = net(valid_data[0])
                pred = torch.argmax(output, dim=1)
                total_accuracy += get_accuracy(pred,valid_data[2])

              accuracy = total_accuracy/len(valid_loader)
              if (epoch > 0) and (accuracy > best_accuracy):
                best_accuracy = accuracy
                best_epoch_weight_path = saveModel(net,epoch)
              print(f'Epoch : [{epoch + 1}/{num_epochs}] | Accuracy: {accuracy} | loss: {epoch_loss}')

  print("Training Finished")

if __name__ == '__main__':

  prepare_dataset()

  train_dataset = Fruit_Dataset('fruits_train.csv','fruits/train')
  valid_dataset = Fruit_Dataset('fruits_validation.csv','fruits/validation')
  test_dataset = Fruit_Dataset('fruits_test.csv','fruits/test')

  print('The shape of tensor for 50th image in train dataset: ',train_dataset[49][0].size())
  print('The class_name for 50th image in train dataset: ',train_dataset[49][1])
  print('The class_id for 50th image in train dataset: ',train_dataset[49][2])

  print('The shape of tensor for 50th image in valid dataset: ',valid_dataset[49][0].size())
  print('The class_name for 50th image in valid dataset: ',valid_dataset[49][1])
  print('The class_id for 50th image in valid dataset: ',valid_dataset[49][2])

  print('The shape of tensor for 50th image in test dataset: ',test_dataset[49][0].size())
  print('The class_name for 50th image in test dataset: ',test_dataset[49][1])
  print('The class_id for 50th image in test dataset: ',test_dataset[49][2])



  batch_size = 16

  train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
                )

  valid_loader = DataLoader(
                valid_dataset, batch_size=batch_size, shuffle=False
                )

  test_loader = DataLoader(
                test_dataset, batch_size =batch_size, shuffle=False
                )

  train()
