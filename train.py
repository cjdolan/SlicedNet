import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import time
from sklearn.metrics import top_k_accuracy_score
import numpy as np

def read_data(data_sample):
  return data_sample[0].cuda(), data_sample[1].cuda()

def train(model, train_loader, val_loader, epochs=300, lr=0.1):
  optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs*0.5),int(epochs*0.75)], gamma=0.1)
  loss_fn = nn.CrossEntropyLoss()
  training_data = pd.DataFrame(columns=['Epoch', 'Train Loss', 'Val Loss', 'Time', 'Train Top-1 Acc', 'Train Top-5 Acc', 'Val Top-1 Acc', 'Val Top-5 Acc'])
  for e in range(epochs):
    model.train()
    tbar = tqdm(train_loader, position=0, leave=True)
    
    start = time.time()
    train_loss_temp = []
    train_acc_1 = []
    train_acc_5 = []
    correct = 0
    total = 0
    for batch, (X, Y) in enumerate(tbar):
      img, label = read_data((X, Y))

      optimizer.zero_grad()
      preds = model(img)
      loss = loss_fn(preds, label)
      loss.backward()
      optimizer.step()

      acc_1 = top_k_accuracy_score(label.detach().cpu().numpy(), preds.detach().cpu().numpy(), k=1, labels = [i for i in range(100)])
      acc_5 = top_k_accuracy_score(label.detach().cpu().numpy(), preds.detach().cpu().numpy(), k=5, labels = [i for i in range(100)])

      _, out = preds.max(1)
      correct += out.eq(label).sum().detach().cpu().numpy()
      total += label.shape[0]
      accuracy = correct / total


      train_acc_1.append(acc_1)
      train_acc_5.append(acc_5)


      train_loss_temp.append(loss.detach().cpu().numpy().ravel())

      tbar.set_description('Epoch: %i, Loss: %f, Top-1: %f, Top-5: %f, Accuracy: %f' % (e+1, np.round(np.mean(train_loss_temp),4), np.round(np.mean(train_acc_1),4), np.round(np.mean(train_acc_5),4), np.round(accuracy, 4)))

    model.eval()
    vbar = tqdm(val_loader, position=0, leave=True)
    val_loss_temp = []
    val_acc_1 = []
    val_acc_5 = []
    with torch.no_grad():
      for batch, (X, Y) in enumerate(vbar):
        img, label = read_data((X, Y))
        preds = model(img)
        loss = loss_fn(preds, label)

        val_loss_temp.append(loss.detach().cpu().numpy().ravel())

        acc_1 = top_k_accuracy_score(label.detach().cpu().numpy(), preds.detach().cpu().numpy(), k=1, labels = [i for i in range(100)])
        acc_5 = top_k_accuracy_score(label.detach().cpu().numpy(), preds.detach().cpu().numpy(), k=5, labels = [i for i in range(100)])

        val_acc_1.append(acc_1)
        val_acc_5.append(acc_5)

        vbar.set_description('Epoch: %i, Val Loss: %f, Top-1: %f, Top-5: %f' % (e+1, np.round(np.mean(val_loss_temp),4), np.round(np.mean(val_acc_1),4), np.round(np.mean(val_acc_5),4)))
    
    end = time.time()
    training_data.at[e, 'Epoch'] = e+1
    training_data.at[e, 'Train Loss'] = np.round(np.mean(train_loss_temp),4)
    training_data.at[e, 'Val Loss'] = np.round(np.mean(val_loss_temp),4)
    training_data.at[e, 'Time'] = end-start
    training_data.at[e, 'Train Top-1 Acc'] = np.round(np.mean(train_acc_1),4)
    training_data.at[e, 'Train Top-5 Acc'] = np.round(np.mean(train_acc_5),4)
    training_data.at[e, 'Val Top-1 Acc'] = np.round(np.mean(val_acc_1),4)
    training_data.at[e, 'Val Top-5 Acc'] = np.round(np.mean(val_acc_5),4)
    scheduler.step()

  return training_data